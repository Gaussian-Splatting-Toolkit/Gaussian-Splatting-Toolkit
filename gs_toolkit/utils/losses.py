import math
import torch
import numpy as np
import cv2
import open3d as o3d


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


def PlaneRegression(
    points: torch.Tensor, threshold: float = 50, init_n: int = 3, iter: int = 1000
):
    """plane regression using ransac

    Args:
        points (torch.Tensor): N x3 point clouds
        threshold (float, optional): distance threshold. Defaults to 0.1.
        init_n (int, optional): Number of initial points to be considered inliers in each iteration
        iter (int, optional): number of iteration. Defaults to 1000.

    Returns:
        [torch.Tensor, List]: 4 x 1 plane equation weights, List of plane point index
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.clone().detach().cpu().numpy())
    w, index = pcd.segment_plane(threshold, init_n, iter)

    return w, index


def plane_distance(points: torch.Tensor, plane: torch.Tensor):
    """Compute the distance of the points to the plane

    Args:
        points (torch.Tensor): N x 3 point clouds
        plane (torch.Tensor): 4 x 1 plane equation weights

    Returns:
        torch.Tensor: N x 1 distance of the points to the plane
    """
    return torch.abs(points @ plane[:3] + plane[3]) / torch.norm(plane[:3])


def planar_loss(
    depth_map: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    """Compute the planar loss

    Args:
        depth_map (torch.Tensor): H x W depth map
        fx (float): focal length in x direction
        fy (float): focal length in y direction
        cx (float): principal point in x direction
        cy (float): principal point in y direction

    Returns:
        torch.Tensor: planar loss
    """
    # Convert depth map to point cloud
    k = torch.eye(4).to(depth_map.device)
    k[0, :3] = torch.FloatTensor([1 / fx, 0, -(cx * fy) / (fx * fy)]).to(
        depth_map.device
    )
    k[1, 1:3] = torch.FloatTensor([1 / fy, -cy / fy]).to(depth_map.device)

    sparse_depth = depth_map.to_sparse()
    indices = sparse_depth.indices()
    values = sparse_depth.values()
    xy_depth = torch.cat((indices.T, values[..., None]), dim=-1)

    final_z = xy_depth[..., -1]

    xy_depth = torch.cat((xy_depth, 1 / xy_depth[..., -1].unsqueeze(-1)), dim=-1)
    xy_depth[:, 2] = 1.0
    points = xy_depth @ k.T * final_z.unsqueeze(-1)
    points = points[:, 0:3]
    # Save the point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points.clone().detach().cpu().numpy())
    # o3d.io.write_point_cloud("debug_full.ply", pcd)

    # Compute the plane equation
    plane, selected_idx = PlaneRegression(points)
    plane = torch.tensor(plane, device="cuda").float()

    # Compute the distance of the points to the plane
    distance = plane_distance(points[selected_idx], plane)

    # Save the segment plane
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[selected_idx].clone().detach().cpu().numpy())
    # o3d.io.write_point_cloud("debug_part.ply", pcd)

    return distance.mean()


def local_planar_loss(depth_src, box_p, fx, fy, cx, cy, ratio=0.5):
    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    num_box_h = math.floor(depth_src.shape[0] / box_p)
    num_box_w = math.floor(depth_src.shape[1] / box_p)
    max_h = max(depth_src.shape[0] - box_p, 0)
    max_w = max(depth_src.shape[1] - box_p, 0)
    _loss = torch.tensor(0.0, device="cuda")
    # Select the number of boxes based on hyperparameter p_corr
    n_corr = int(ratio * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device="cuda")
    y_0 = torch.randint(0, max_w, size=(n_corr,), device="cuda")
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0, device="cuda")
    for i in range(len(x_0)):
        _loss += planar_loss(
            depth_src[x_0[i] : x_1[i], y_0[i] : y_1[i]], fx, fy, cx, cy
        )
    return _loss / n_corr
