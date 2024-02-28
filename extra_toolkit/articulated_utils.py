import numpy as np
import open3d as o3d

def rotate_and_combine(angle, total_angle, axis_of_rotation, cuff_point_cloud, rest_point_cloud):
    # Find the center of rotation (assuming it's the connection point between the cuff and the rest)
    # This could be an average of some points or a specific point you define
    center_of_rotation = np.array([(a + b) / 2 for a, b in zip(end_1, end_2)])  # Replace x, y, z with the coordinates of the connection point

    R = cuff_point_cloud.get_rotation_matrix_from_axis_angle(axis_of_rotation * np.radians(angle))
    cuff_point_cloud.rotate(R, center=center_of_rotation)
    
    # Combine the two point clouds
    combined_point_cloud = cuff_point_cloud + rest_point_cloud
    o3d.io.write_point_cloud(base_path + f"interim_rotate_angle_{total_angle}.ply", combined_point_cloud)

if __name__ == "__main__":
    base_path = "/mnt/d/Projects/reconstruction/result/small_obj/orange_clothes/move/stage_3/"

    # Load the point clouds (assuming they are in .ply format)
    cuff_point_cloud = o3d.io.read_point_cloud(base_path + 'first_half.ply')
    rest_point_cloud = o3d.io.read_point_cloud(base_path + 'second_half.ply')

    # Define the axis of rotation and angle (in radians)
    # For example, rotating around the z-axis by 45 degrees
    end_1 = [-0.150391, -0.400391, -0.878532]
    end_2 = [0.384766, -0.0917969, -0.877352]
    axis_of_rotation = [a - b for a, b in zip(end_1, end_2)]
    
    # Normalize the axis of rotation
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)
    
    angle = 10
    for total_angle in range(0, 180, angle):
        print(f"Rotating by {total_angle} degrees")
        rotate_and_combine(angle, total_angle, axis_of_rotation, cuff_point_cloud, rest_point_cloud)