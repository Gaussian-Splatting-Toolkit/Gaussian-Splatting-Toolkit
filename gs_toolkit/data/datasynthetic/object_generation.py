import json
import blendersynth as bsyn
import numpy as np
import os
import open3d as o3d

bsyn.run_this_script()
bsyn.render.render_with_gpu()

mesh_path = "data/meshes/movo_body/movo_upper_body.obj"
output_folder = "data/movo"
# Create colmap folder in the output folder
os.makedirs(output_folder + "/colmap", exist_ok=True)

# Generate the point cloud from mesh
o3d_mesh = o3d.io.read_triangle_mesh(mesh_path, True)
points = o3d_mesh.sample_points_poisson_disk(100000)
# Save the point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.points)
o3d.io.write_point_cloud(output_folder + "/colmap/point_cloud.ply", pcd)
print("Point cloud saved")

include_depth = True

# monkey = bsyn.Mesh.from_primitive("monkey")
mesh = bsyn.Mesh.from_obj(mesh_path)
# mesh.material = bsyn.Material.add_source("data/meshes/movo_body/movo_upper_body.mtl")
bsyn.world.set_color((0.8, 0.7, 0.8))

# Get the largest dimension of the mesh
max_dim = max(mesh.dimensions)

w, h = 1280, 720

bsyn.render.set_resolution(w, h)
bsyn.render.set_cycles_samples(100)

comp = bsyn.Compositor()
obj_pass_idx = mesh.assign_pass_index(
    1
)  # To show masking, we assign a pass index to the first object

# we will create 4 cameras, one from each side of the monkey, facing the monkey
cameras = []
extrinsics = []
names = []
camera_radius = 5 * max_dim
# Generate 8 layers of cameras, each layer has 20 cameras, looking at the mesh from different angles
no_layers = 8
no_rot_poses = 20
vertical_angle = np.pi / (no_layers + 2)
horizontal_angle = 2 * np.pi / no_rot_poses

for i in range(1, no_layers + 1):
    for j in range(no_rot_poses):
        name = f"Cam{i}_{j}"
        camera = bsyn.Camera.create(
            name=name,
            location=(
                camera_radius
                * np.cos(j * horizontal_angle)
                * np.sin(i * vertical_angle),
                camera_radius
                * np.sin(j * horizontal_angle)
                * np.sin(i * vertical_angle),
                camera_radius * np.cos(i * vertical_angle),
            ),
        )
        camera.look_at_object(mesh)
        cameras.append(camera)

        light = bsyn.Light.create("POINT", location=camera.location, intensity=3)

        # Save the extrinsics for each camera
        extrinsics.append(np.array(camera.matrix_world))
        names.append(name)

bsyn.render.render_depth()  # Enable standard Blender depth pass
depth_vis = comp.get_depth_visual(max_depth=20)  # Create a visual of the depth pass
rgb_mask = comp.get_mask(
    obj_pass_idx, "Image"
)  # create an RGB mask (i.e. only render monkey)
bounding_box_visual = comp.get_bounding_box_visual()
# keypoints_visual = comp.get_keypoints_visual()  # Create a visual of keypoints


# we'll render RGB, normals, and bounding boxes
normal_aov = bsyn.aov.NormalsAOV(polarity=[-1, 1, -1])
instancing_aov = bsyn.aov.InstanceRGBAOV()
class_aov = bsyn.aov.ClassRGBAOV()
UVAOV = bsyn.aov.UVAOV()  # UV Coordinates
NOCAOV = bsyn.aov.GeneratedAOV()  # Normalized Object Coordinates (NOC)

for aov in [normal_aov, instancing_aov, class_aov, UVAOV, NOCAOV]:
    mesh.assign_aov(aov)

os.makedirs(output_folder, exist_ok=True)

# All of the following will have Blender's Filmic colour correction by default
comp.define_output("Image", file_name="rgb", directory=output_folder + "/rgb")
comp.define_output(
    rgb_mask, name="rgb_masked", directory=output_folder + "/rgb_masked"
)  # render RGB layer masked by monkey
comp.define_output(
    bounding_box_visual, output_folder + "/bounding_box", name="bounding_box_visual"
)
# comp.define_output(keypoints_visual, output_folder, name='keypoints')

# All of the following will not have any colour correction
comp.define_output(
    normal_aov, file_name="normals", directory=output_folder + "/normals"
)
comp.define_output(
    depth_vis, output_folder + "/depth_vis"
)  # render visual of depth layer
comp.define_output(instancing_aov, output_folder + "/instancing", name="instancing")
comp.define_output(class_aov, output_folder + "/semantic", name="semantic")
comp.define_output(UVAOV, output_folder + "/UV", name="UV")
comp.define_output(NOCAOV, output_folder + "/NOC", name="NOC")
comp.define_output("Depth", output_folder + "/depth", file_format="OPEN_EXR")

# Generate the transform.json
# extrinsics = [extrinsic.tolist() for extrinsic in extrinsics]
focal_length = camera.focal_length
sensor_width = camera.data.sensor_width
sensor_height = camera.data.sensor_height
pixel_aspect_ratio = (
    bsyn.context.scene.render.pixel_aspect_x / bsyn.context.scene.render.pixel_aspect_y
)
s_u = w / sensor_width
s_v = h * pixel_aspect_ratio / sensor_height
cx = w / 2
cy = h / 2

data = {}

data["w"] = w
data["h"] = h
data["fl_x"] = focal_length * s_u
data["fl_y"] = focal_length * s_v
data["cx"] = cx
data["cy"] = cy
data["k1"] = 0
data["k2"] = 0
data["p1"] = 0
data["p2"] = 0
data["camera_model"] = "OPENCV"
data["ply_file_path"] = "colmap/point_cloud.ply"
data["applied_scale"] = 1.0

frames = []
for i, name in enumerate(names):
    frame = {}
    frame["file_path"] = "rgb_masked/" + name + "_rgb_masked.png"
    if include_depth:
        frame["depth_file_path"] = "depth/" + name + "_Depth.exr"
    # c2w = np.linalg.inv(extrinsics[i])
    c2w = extrinsics[i]
    # Convert from Blender's camera coordinates to ours (OpenGL)
    c2w = c2w[np.array([0, 2, 1, 3]), :]
    c2w[2, :] *= -1
    frame["transform_matrix"] = c2w.tolist()
    frame["colmap_id"] = i
    frames.append(frame)

data["frames"] = frames

with open(output_folder + "/transforms.json", "w") as f:
    json.dump(data, f)

bounding_boxes = bsyn.annotations.bounding_boxes([mesh], cameras)
# keypoints = bsyn.annotations.keypoints.project_keypoints(cube_vertices)
comp.render(camera=cameras, annotations=bounding_boxes)
