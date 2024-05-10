"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from rasterizer.project_gaussians import project_gaussians
from rasterizer.rasterize import rasterize_gaussians
from rasterizer.sh import spherical_harmonics

from gs_toolkit.cameras.camera_optimizers import CameraOptimizerConfig
from gs_toolkit.cameras.cameras import Cameras

# need following import for background color override
from gs_toolkit.model_components import renderers

from gs_toolkit.utils.comms import projection_matrix
from gs_toolkit.models.vanilla_gs import (
    GaussianSplattingModelConfig,
    GaussianSplattingModel,
)

from gs_toolkit.utils.losses import (
    local_pearson_loss,
    l2_loss,
    nearMean_map,
    image2canny,
    local_planar_loss,
    tv_Loss,
)


@dataclass
class DepthGSModelConfig(GaussianSplattingModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: DepthGSModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 2000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 8000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    depth_lambda: float = 0.2
    """weight of depth loss"""
    stop_split_at: int = 25_000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=CameraOptimizerConfig
    )
    """camera optimizer config"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    use_sparse_loss: bool = False
    """If True, use sparse loss for alpha channel"""
    sparse_lambda: float = 0.1
    """weight of sparse loss"""
    use_depth_loss: bool = True
    """If True, use depth loss"""
    depth_lambda: float = 0.1
    """weight of depth loss"""
    depth_loss_start_iteration: int = 6_000
    """start iteration of depth loss"""
    depth_loss_stop_iteration: int = 25_000
    """stop iteration of depth loss"""
    use_est_depth: bool = False
    """If True, use estimated depth for depth loss"""
    use_pearson_depth: bool = False
    """If True, use pearson depth loss"""
    mono_depth_l1_start_iteration: int = 15_000
    """start iteration of mono depth l1 loss"""
    use_scaled_est_depth: bool = False
    """If True, use scaled estimated depth for depth loss"""
    local_patch_size: int = 128
    """size of local patch for local pearson loss"""
    use_depth_regularization: bool = False
    """If True, use depth regularization"""
    using_planar_loss: bool = False
    """If True, use planar loss"""
    planar_loss_start_iteration: int = 10_000
    """start iteration of planar loss"""
    using_tv_loss: bool = False
    """If True, use total variation loss"""


class DepthGSModel(GaussianSplattingModel):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: DepthGSModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # self.seed_points = seed_points
        super().__init__(seed_points=seed_points, *args, **kwargs)

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            if len(image.shape) == 2:
                # Squeeze the dimension to 3 and then unsqueeze it back to 2
                return TF.resize(image[None, :, :], newsize, antialias=None)[0]
            else:
                return TF.resize(
                    image.permute(2, 0, 1), newsize, antialias=None
                ).permute(1, 2, 0)
        return image

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(
                    int(camera.height.item()), int(camera.width.item()), 1
                )
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {
                    "rgb": rgb,
                    "depth": depth,
                    "accumulation": accumulation,
                    "background": background,
                }
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with rasterizer conventions
        R_edit = torch.diag(
            torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype)
        )
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )
        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )
        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {
                "rgb": rgb,
                "depth": depth,
                "accumulation": accumulation,
                "background": background,
            }

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = (
                means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]
            )  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        rgb, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[
                ..., 0:1
            ]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        return {"rgb": rgb, "depth": depth_im, "background": background}  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        if "depth" in batch and self.step > self.config.depth_loss_start_iteration:
            gt_depth = self.get_gt_img(batch["depth"])
        pred_img = outputs["rgb"]
        pred_depth = outputs["depth"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"]).unsqueeze(-1)
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            rgb_mask = mask.repeat(1, 1, 3)
            gt_img = gt_img * rgb_mask
            pred_img = pred_img * rgb_mask

            if self.step > self.config.depth_loss_start_iteration:
                gt_depth = gt_depth * mask
            pred_depth = pred_depth * mask

        loss_dict = {}
        # local_size = [256, 128, 64]
        # selected_size = local_size[((self.step - self.config.depth_loss_start_iteration) // 1000) % 3]

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )
        loss_dict["main_loss"] = (1 - self.config.ssim_lambda) * Ll1
        +self.config.ssim_lambda * simloss

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        if self.config.use_sparse_loss and self.step % 100 == 0:
            l_sparse = (
                torch.log(self.gauss_params["opacities"] + 1e-6)
                + torch.log(1 - self.gauss_params["opacities"] + 1e-6)
            ).mean()
            loss_dict["sparse_loss"] = self.config.sparse_lambda * l_sparse

        loss_dict["scale_reg"] = scale_reg

        if (
            "depth" in batch
            and self.config.use_depth_loss
            and self.step > self.config.depth_loss_start_iteration
        ):
            if self.config.use_est_depth:
                if (
                    self.step < self.config.depth_loss_stop_iteration
                    and self.config.use_pearson_depth
                ):
                    loss_dict["depth_local_pearson"] = local_pearson_loss(
                        pred_depth, gt_depth, self.config.local_patch_size, 0.5
                    )
                    # loss_dict["depth_global_pearson"] = pearson_depth_loss(
                    #     pred_depth.reshape(-1), gt_depth.reshape(-1)
                    # )
                pred_depth = pred_depth.squeeze(-1)
                canny_mask = (
                    image2canny(gt_img, 50, 150, isEdge1=False).detach().to(self.device)
                )
                if self.config.use_scaled_est_depth:
                    if "mono_depth_scale" in batch:
                        scaled_pred_depth = (
                            batch["mono_depth_scale"] * pred_depth
                            + batch["mono_depth_shift"]
                        )
                        logl1 = torch.log(1 + torch.abs(gt_depth - scaled_pred_depth))

                        grad_img_x = torch.mean(
                            torch.abs(gt_img[:, :-1, :] - gt_img[:, 1:, :]),
                            -1,
                            keepdim=True,
                        )
                        grad_img_y = torch.mean(
                            torch.abs(gt_img[:-1, :, :] - gt_img[1:, :, :]),
                            -1,
                            keepdim=True,
                        )
                        lambda_x = torch.exp(-grad_img_x).squeeze(-1)
                        lambda_y = torch.exp(-grad_img_y).squeeze(-1)

                        loss_x = lambda_x * logl1[:, :-1]
                        loss_y = lambda_y * logl1[:-1, :]

                        # loss_x = loss_x[canny_mask[:, :-1]]
                        # loss_y = loss_y[canny_mask[:-1, :]]

                        loss_dict["log_depth"] = loss_x.mean() + loss_y.mean()

                if self.config.use_depth_regularization:
                    depth_mask = (pred_depth > 0).detach()
                    nearDepthMean_map = nearMean_map(
                        pred_depth, canny_mask * depth_mask
                    )
                    loss_dict["depth_reg_loss"] = (
                        l2_loss(nearDepthMean_map, pred_depth * depth_mask) * 1.0
                    )

                if self.config.using_tv_loss and self.step < 20_000:
                    loss_dict["tv_loss"] = tv_Loss(pred_depth)
            else:
                depth_nonzero = gt_depth > 0
                pred_depth = pred_depth.squeeze(-1)
                Ll1_depth = torch.abs(
                    gt_depth * depth_nonzero - pred_depth * depth_nonzero
                ).mean()
                loss_dict["depth_l1"] = Ll1_depth

        return loss_dict

    def add_planar_loss(self, cameras: Cameras, loss_dict, output):
        if (
            self.step > self.config.planar_loss_start_iteration
            and self.config.using_planar_loss
        ):
            loss_dict["planar_loss"] = 10 * local_planar_loss(
                output["depth"],
                self.config.local_patch_size,
                cameras.fx,
                cameras.fy,
                cameras.cx,
                cameras.cy,
            )
