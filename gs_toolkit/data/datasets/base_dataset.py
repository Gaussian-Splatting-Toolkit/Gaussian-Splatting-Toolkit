"""
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from gs_toolkit.cameras.cameras import Cameras
from gs_toolkit.data.dataparsers.base_dataparser import DataparserOutputs
from gs_toolkit.data.utils.data_utils import (
    get_image_mask_tensor_from_path,
    exr_to_array,
)


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask", "depth"]
    cameras: Cameras

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0
    ):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(
        self, image_idx: int
    ) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(
            self.get_numpy_image(image_idx).astype("float32") / 255.0
        )
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            image = image[:, :, :3] * image[
                :, :, -1:
            ] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_numpy_depth_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        depth_filename = self._dataparser_outputs.metadata["depth_filenames"][image_idx]
        # If the file end with exr
        if depth_filename.endswith(".exr"):
            depth = exr_to_array(depth_filename)
        else:
            pil_image = Image.open(depth_filename)
            if self.scale_factor != 1.0:
                width, height = pil_image.size
                newsize = (
                    int(width * self.scale_factor),
                    int(height * self.scale_factor),
                )
                pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
            depth = np.array(pil_image)  # shape is (h, w) or (h, w, 3 or 4)
        # assert len(depth.shape) == 3
        # assert depth.dtype == np.uint8
        # assert depth.shape[2] == 1, f"Image shape of {depth.shape} is in correct."
        return depth

    def get_depth_image(
        self, depth_idx: int
    ) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 1 channel depth map.

        Args:
            depth_idx: The depth image index in the dataset.
        """
        if self._dataparser_outputs.metadata["mono_depth_scales"] is not None:
            depth = torch.from_numpy(
                self.get_numpy_depth_image(depth_idx).astype("float32") / 255.0
            )
        else:
            depth = torch.from_numpy(
                self.get_numpy_depth_image(depth_idx).astype("float32") / 1000.0
            )
        return depth

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        if self._dataparser_outputs.metadata["depth_filenames"] is not None:
            depth = self.get_depth_image(image_idx)
            data = {"image_idx": image_idx, "image": image, "depth": depth}
        else:
            data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(
                filepath=mask_filepath, scale_factor=self.scale_factor
            )
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if len(self._dataparser_outputs.metadata["mono_depth_scales"]) != 0:
            data["mono_depth_scale"] = self._dataparser_outputs.metadata[
                "mono_depth_scales"
            ][image_idx]
        if len(self._dataparser_outputs.metadata["mono_depth_shifts"]) != 0:
            data["mono_depth_shift"] = self._dataparser_outputs.metadata[
                "mono_depth_shifts"
            ][image_idx]
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames
