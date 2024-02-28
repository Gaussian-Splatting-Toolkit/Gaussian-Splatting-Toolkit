import argparse
import json
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm


mesh_extract_methods = ["marching_cubes", "poisson"]


class CameraPose:
    def __init__(self, meta, mat) -> None:
        self.metadata = meta
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


class RGBDFusion:
    def __init__(
        self,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        data_path: str,
        save_path: str = None,
        using_OpenGL_world_coord: bool = False,
        method: str = "marching_cubes",
        experiment_name: str = "experiment",
        mask: bool = False,
        device: str = "cuda:0",
        filter_ply: bool = False,
        bounding_box: bool = True,
    ) -> None:
        self.intrinsic = intrinsic
        self.data_path = data_path
        self.save_path = save_path
        self.use_OpenGL_world_coord = using_OpenGL_world_coord
        self.method = method
        self.camera_poses_path = self.data_path + "/poses.json"
        self.camera_poses = self.read_trajectory()
        self.experiment_name = experiment_name
        self.mask = mask
        self.device = o3d.core.Device(device)
        self.filter_ply = filter_ply
        self.bounding_box = bounding_box

    def read_trajectory(self) -> list[CameraPose]:
        traj = []
        # Read from json file
        f = open(self.camera_poses_path, "r")
        data = json.load(f)
        for idx, pose in enumerate(data["camera_path"]):
            metadata = idx
            mat = np.array(pose)
            if self.use_OpenGL_world_coord:
                mat[2, :] *= -1
                mat = mat[np.array([1, 0, 2, 3]), :]
                mat[0:3, 1:3] *= -1
            transform = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            mat = transform @ mat
            traj.append(CameraPose(metadata, mat))
        return traj

    def integrate(self) -> o3d.pipelines.integration.ScalableTSDFVolume:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, camera_pose in tqdm(
            enumerate(self.camera_poses),
            desc="Integrating",
            total=len(self.camera_poses),
        ):
            color = o3d.io.read_image(
                self.data_path + f"/rgb/frame_{i:05}.png"
            )
            depth = o3d.io.read_image(
                self.data_path + f"/depth/depth_{i:05}.png"
            )
            if self.mask:
                mask = o3d.io.read_image(
                    self.data_path + f"/mask/frame_{i:05}.png"
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
            volume.integrate(rgbd, self.intrinsic, np.linalg.inv(camera_pose.pose))

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
            if self.filter_ply:
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
        for i, camera_pose in tqdm(
            enumerate(self.camera_poses), desc="Filtering", total=len(self.camera_poses)
        ):
            # Project all points onto the 2D camera plane
            points_2d = self.batch_project_to_2d(
                points, self.intrinsic, np.linalg.inv(camera_pose.pose)
            )

            # Load the corresponding mask
            mask = o3d.io.read_image(
                self.data_path + f"/mask/rgb_{i}.png"
            )
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
        new_mask[y_min:y_max+1, x_min:x_max+1] = 1

        return new_mask

    def run(self) -> None:
        volume = self.integrate()
        mesh, pcd = self.extract_mesh(volume)
        if self.save_path is not None:
            # Save point cloud to ply
            os.makedirs(self.save_path + "/" + self.experiment_name, exist_ok=True)
            o3d.io.write_point_cloud(
                self.save_path + "/" + self.experiment_name + "/point_cloud.ply", pcd
            )
            print("Extract a point cloud with {:d} points.".format(len(pcd.points)))
            # Save mesh
            o3d.io.write_triangle_mesh(
                self.save_path
                + "/"
                + self.experiment_name
                + f"/mesh_{self.method}.ply",
                mesh,
            )
            print("Extract a mesh with {:d} triangles.".format(len(mesh.triangles)))


def main():
    # Argument parser
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_path", type=str, default="data/test_apple", help="path to data")
    arg.add_argument(
        "--method",
        type=str,
        default="marching_cubes",
        choices=mesh_extract_methods,
        help="mesh extract method, choose from [marching_cubes, poisson]",
    )
    arg.add_argument(
        "--save_path", type=str, default="result", help="path to save result"
    )
    arg.add_argument(
        "--use_OpenGL_world_coord",
        action="store_true",
        help="if using OpenGL coord, we need to convert it to Open3D coord",
    )
    arg.add_argument(
        "--experiment_name", type=str, default="experiment", help="experiment name"
    )
    arg.add_argument(
        "--mask", action="store_true", help="if using mask to extract mesh"
    )
    arg.add_argument(
        "--device", type=str, default="cuda:0", help="device to run the code"
    )
    arg.add_argument(
        "--add_noise", action="store_true", help="if adding noise to camera pose"
    )
    arg.add_argument(
        "--filter_point", action="store_true", help="if filtering point cloud"
    )
    args = arg.parse_args()

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=2160,
        height=3840,
        fx=617.5422957061793,
        fy=614.7499779868493,
        cx=644.4233510471437,
        cy=357.7418022307792,
    )

    fusion = RGBDFusion(
        intrinsic,
        args.data_path,
        args.save_path,
        args.use_OpenGL_world_coord,
        args.method,
        args.experiment_name,
        args.mask,
        args.device,
        args.filter_point,
    )
    fusion.run()


if __name__ == "__main__":
    main()
