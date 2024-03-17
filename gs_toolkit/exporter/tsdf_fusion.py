from pathlib import Path
import open3d as o3d
import numpy as np
import json
from rich.progress import track


class CameraPose:
    def __init__(self, meta, mat, intrinsic) -> None:
        self.metadata = meta
        self.intrinsic = intrinsic
        self.pose = mat

    def __str__(self) -> str:
        return (
            "Metadata : "
            + " ".join(map(str, self.metadata))
            + "\n"
            + "Pose : "
            + "\n"
            + np.array_str(self.pose)
        )


class TSDFFusion:
    def __init__(
        self,
        data_path: Path,
        using_OpenGL_world_coord: bool = False,
        method: str = "marching_cubes",
        voxel_length: float = 4.0 / 512.0,
        sdf_trunc: float = 0.04,
        mask: bool = False,
        filter_pcd: bool = False,
        bounding_box: bool = False,
        using_gt: bool = False,
    ) -> None:
        self.data_path = data_path
        self.use_OpenGL_world_coord = using_OpenGL_world_coord
        self.method = method
        self.camera_poses_path = self.data_path / "poses.json"
        self.camera_poses = self.read_trajectory()
        self.mask = mask
        self.filter_pcd = filter_pcd
        self.bounding_box = bounding_box
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.using_gt = using_gt

    def read_trajectory(self) -> list[CameraPose]:
        traj = []
        # Read from json file
        f = open(self.camera_poses_path, "r")
        data = json.load(f)
        for idx, camera in enumerate(data):
            metadata = idx
            mat = np.array(camera["pose"])
            if self.use_OpenGL_world_coord:
                mat[2, :] *= -1
                mat = mat[np.array([1, 0, 2, 3]), :]
                mat[0:3, 1:3] *= -1
            transform = np.array(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            mat = transform @ mat
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=camera["camera"]["width"],
                height=camera["camera"]["height"],
                fx=camera["camera"]["fx"],
                fy=camera["camera"]["fy"],
                cx=camera["camera"]["cx"],
                cy=camera["camera"]["cy"],
            )
            traj.append(CameraPose(metadata, mat, intrinsic))
        return traj

    def integrate(self) -> o3d.pipelines.integration.ScalableTSDFVolume:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, camera_pose in track(
            enumerate(self.camera_poses),
            description="Integrating",
            total=len(self.camera_poses),
        ):
            if self.using_gt:
                color_path = self.data_path / "gt" / "rgb" / f"frame_{i:05}.jpg"
                color = o3d.io.read_image(str(color_path))
                depth_path = self.data_path / "gt" / "depth" / f"depth_{i:05}.png"
                depth = o3d.io.read_image(str(depth_path))
                # Modify the intrinsic width and height based on the color image
                camera_pose.intrinsic.set_intrinsics(
                    width=np.array(color).shape[1],
                    height=np.array(color).shape[0],
                    fx=camera_pose.intrinsic.intrinsic_matrix[0, 0],
                    fy=camera_pose.intrinsic.intrinsic_matrix[1, 1],
                    cx=camera_pose.intrinsic.intrinsic_matrix[0, 2],
                    cy=camera_pose.intrinsic.intrinsic_matrix[1, 2],
                )
            else:
                color = o3d.io.read_image(
                    str(self.data_path / "rgb" / f"frame_{i:05}.png")
                )
                depth = o3d.io.read_image(
                    str(self.data_path / "depth" / f"depth_{i:05}.png")
                )
            if self.mask:
                mask = o3d.io.read_image(
                    str(self.data_path / "mask" / f"frame_{i:05}.png")
                )
                # Set depth to 0 where mask is 0
                mask_np = np.asarray(mask)
                if self.bounding_box:
                    # Get the bounding box of the mask
                    bb_mask = self.create_bounding_box_mask(mask_np, 5)
                    depth_np = np.asarray(depth)
                    masked_depth_np = depth_np * bb_mask
                    depth = o3d.geometry.Image(np.uint16(masked_depth_np))
                else:
                    # Select mask where mask is not 0
                    mask_np[mask_np > 0] = 1
                    depth_np = np.asarray(depth)
                    masked_depth_np = depth_np * mask_np
                    depth = o3d.geometry.Image(np.uint16(masked_depth_np))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False
            )
            volume.integrate(
                rgbd, camera_pose.intrinsic, np.linalg.inv(camera_pose.pose)
            )

        return volume

    def extract_mesh(
        self, volume: o3d.pipelines.integration.ScalableTSDFVolume
    ) -> list[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
        if self.method == "marching_cubes":
            mesh = volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            pcd = volume.extract_point_cloud()
            return mesh, pcd
        elif self.method == "poisson":
            if self.filter_pcd:
                pcd = self.filter_point(volume)
            else:
                pcd = volume.extract_point_cloud()
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=11
            )
            mesh.compute_vertex_normals()
            return mesh, pcd
        else:
            raise NotImplementedError

    def batch_project_to_2d(self, points_3d, intrinsic, pose):
        """
        Batch projects 3D points onto the 2D camera plane.
        """
        # Transform points using camera pose
        points_transformed = (pose[:3, :3] @ points_3d.T).T + pose[:3, 3]

        # Project points using the camera intrinsic matrix
        points_projected = (intrinsic.intrinsic_matrix[:3, :3] @ points_transformed.T).T
        points_2d = points_projected[:, :2] / points_projected[:, 2:]

        return points_2d

    def batch_filter_points(self, points_2d, mask):
        """
        Filters the points based on the mask in a batch operation.
        """
        x, y = points_2d.T
        valid_indices = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])

        # Create an array to hold the final results
        final_validity = np.zeros_like(valid_indices, dtype=bool)

        # Ensure indices are integers for mask indexing
        x_int, y_int = x[valid_indices].astype(int), y[valid_indices].astype(int)

        # Get mask values at the projected 2D points
        mask_values = mask[y_int, x_int]

        # Update the final_validity array for valid indices
        final_validity[valid_indices] = mask_values != 0

        return final_validity

    def filter_point(self, volume) -> o3d.geometry.PointCloud:
        # After integrating, extract 3D points from the volume
        pcd = volume.extract_point_cloud()
        points = np.asarray(pcd.points)

        valid_points = np.zeros(len(points), dtype=int)

        # Process each camera pose and update the valid_points array
        for i, camera_pose in track(
            enumerate(self.camera_poses),
            description="Filtering",
            total=len(self.camera_poses),
        ):
            # Project all points onto the 2D camera plane
            points_2d = self.batch_project_to_2d(
                points, self.intrinsic, np.linalg.inv(camera_pose.pose)
            )

            # Load the corresponding mask
            mask = o3d.io.read_image(self.data_path + f"/mask/rgb_{i}.png")
            mask_np = np.asarray(mask)
            mask_np[mask_np > 0] = 1

            # Update the valid_points array
            valid_points += self.batch_filter_points(points_2d, mask_np)

        # Filter the points based on the valid_points array
        valid_points[valid_points <= len(self.camera_poses) // 2] = 0
        valid_points = valid_points.astype(bool)
        filtered_points = points[valid_points]
        filtered_colors = np.asarray(pcd.colors)[valid_points]
        filtered_normals = np.asarray(pcd.normals)[valid_points]

        # Create a new point cloud with the filtered points
        filtered_point_cloud = o3d.geometry.PointCloud()
        filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
        filtered_point_cloud.normals = o3d.utility.Vector3dVector(filtered_normals)
        return filtered_point_cloud

    def create_bounding_box_mask(self, original_mask, margin):
        """
        Create a new mask that is a bounding box covering the original mask with a specified margin.

        Parameters:
        - original_mask: A 2D NumPy array representing the original mask.
        - margin: An integer representing the margin around the bounding box.

        Returns:
        - A 2D NumPy array representing the new mask with the bounding box.
        """
        # Find the coordinates of non-zero elements in the original mask
        nonzero_indices = np.argwhere(original_mask)

        # Get the minimum and maximum coordinates along each axis
        y_min, x_min = np.min(nonzero_indices, axis=0)
        y_max, x_max = np.max(nonzero_indices, axis=0)

        # Add margin to the bounding box
        y_min = max(y_min - margin, 0)
        y_max = min(y_max + margin, original_mask.shape[0] - 1)
        x_min = max(x_min - margin, 0)
        x_max = min(x_max + margin, original_mask.shape[1] - 1)

        # Create a new mask with the bounding box
        new_mask = np.zeros_like(original_mask)
        new_mask[y_min : y_max + 1, x_min : x_max + 1] = 1

        return new_mask

    def run(self) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
        volume = self.integrate()
        mesh, pcd = self.extract_mesh(volume)
        return mesh, pcd
