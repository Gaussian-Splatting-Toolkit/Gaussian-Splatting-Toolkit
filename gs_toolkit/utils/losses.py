import math
import torch
import numpy as np
import cv2


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def pearson_depth_loss(depth_src, depth_target):
    # Compute mean of the depth values
    mean_src = torch.mean(depth_src)
    mean_target = torch.mean(depth_target)
    # Compute the covariance
    cov = torch.mean((depth_src - mean_src) * (depth_target - mean_target))
    # Compute the standard deviations
    std_src = torch.std(depth_src)
    std_target = torch.std(depth_target)
    # Compute the Pearson correlation coefficient
    pearson_corr = cov / (std_src * std_target)
    return 1 - pearson_corr


def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    num_box_h = math.floor(depth_src.shape[0] / box_p)
    num_box_w = math.floor(depth_src.shape[1] / box_p)
    max_h = max(depth_src.shape[0] - box_p, 0)
    max_w = max(depth_src.shape[1] - box_p, 0)
    _loss = torch.tensor(0.0, device="cuda")
    # Select the number of boxes based on hyperparameter p_corr
    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device="cuda")
    y_0 = torch.randint(0, max_w, size=(n_corr,), device="cuda")
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0, device="cuda")
    for i in range(len(x_0)):
        _loss += pearson_depth_loss(
            depth_src[x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1),
            depth_target[x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1),
        )
    return _loss / n_corr


def image2canny(image, thres1, thres2, isEdge1=True):
    """image: (H, W, 3)"""
    canny_mask = torch.from_numpy(
        cv2.Canny(
            (image.detach().cpu().numpy() * 255.0).astype(np.uint8), thres1, thres2
        )
        / 255.0
    )
    if not isEdge1:
        canny_mask = 1.0 - canny_mask
    return canny_mask.float()


with torch.no_grad():
    kernelsize = 3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize // 2))
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]).reshape(
        1, 1, kernelsize, kernelsize
    )
    conv.weight.data = kernel  # torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.0])
    conv.requires_grad_(False)
    conv = conv.cuda()


def nearMean_map(array, mask):
    """array: (H,W) / mask: (H,W)"""
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None, None])
    cnt_map = conv((cnt_map * mask)[None, None])
    nearMean_map = (nearMean_map / (cnt_map + 1e-8)).squeeze()

    return nearMean_map
