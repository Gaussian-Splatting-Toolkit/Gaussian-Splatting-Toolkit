import bpy
import os
# from mathutils import Matrix
from math import pi, sin, cos

# Output directory
output_dir = '/data/synthetic/shpere_3'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Clear existing objects and materials
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create a sphere with smooth shading and reflective material
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
sphere = bpy.context.active_object
bpy.ops.object.shade_smooth()

mat = bpy.data.materials.new(name="ReflectiveMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
for n in nodes:
    if n.type != 'OUTPUT_MATERIAL':  # Keep the material output node
        nodes.remove(n)
node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
node_bsdf.inputs['Metallic'].default_value = 0.5
node_output = nodes.get('Material Output')
mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
sphere.data.materials.append(mat)

# Set up the cameras in a semi-circle facing the object
camera_data = bpy.data.cameras.new("CameraData")
camera_data.lens = 50

num_cameras = 8
for i in range(num_cameras):
    angle = pi * 2 * i / num_cameras
    cam_location = (sin(angle) * 3, cos(angle) * 3, 1)
    bpy.ops.object.camera_add(location=cam_location)
    cam = bpy.context.active_object
    cam.data = camera_data
    cam.rotation_euler = (pi/2, 0, -angle)

# Render settings
scene = bpy.context.scene
scene.use_nodes = True
tree = scene.node_tree
bpy.context.view_layer.use_pass_z = True
bpy.context.view_layer.use_pass_normal = True

# Clear default nodes
for node in tree.nodes:
    tree.nodes.remove(node)
    

# Create a render layers node
render_layers = tree.nodes.new('CompositorNodeRLayers')

# Create output nodes for the render layers
outputs = {}

# RGB Image Output Node
image_file_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
image_file_output_node.base_path = output_dir
image_file_output_node.format.file_format = 'PNG'
image_file_output_node.file_slots[0].path = 'rgb_'
outputs['Image'] = image_file_output_node

# Depth Output Node (scaled and converted to PNG)
depth_file_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output_node.base_path = output_dir
depth_file_output_node.format.file_format = 'PNG'
depth_file_output_node.file_slots[0].path = 'depth_'
outputs['Depth'] = depth_file_output_node

# Normal Output Node (converted to PNG)
normal_file_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output_node.base_path = output_dir
normal_file_output_node.format.file_format = 'PNG'
normal_file_output_node.file_slots[0].path = 'normal_'
outputs['Normal'] = normal_file_output_node

# Link nodes to render layers
tree.links.new(render_layers.outputs['Image'], outputs['Image'].inputs[0])

# For depth, we need to normalize it because PNG can't store the full range like OpenEXR can
depth_normalize_node = tree.nodes.new(type="CompositorNodeNormalize")
tree.links.new(render_layers.outputs['Depth'], depth_normalize_node.inputs['Value'])
tree.links.new(depth_normalize_node.outputs['Value'], outputs['Depth'].inputs[0])

# For normals, we need to encode it from [-1, 1] to [0, 1] range
normal_map_node = tree.nodes.new(type="CompositorNodeMapValue")
# Set the offset and size to map [-1, 1] to [0, 1]
normal_map_node.offset = [0.5, 0.5, 0.5]
normal_map_node.size = [0.5, 0.5, 0.5]
# Prevent clipping
normal_map_node.use_min = True
normal_map_node.min = [0, 0, 0]
normal_map_node.use_max = True
normal_map_node.max = [1, 1, 1]
tree.links.new(render_layers.outputs['Normal'], normal_map_node.inputs[0])
tree.links.new(normal_map_node.outputs[0], outputs['Normal'].inputs[0])

# Set resolution
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# Render from each camera and save
for index, obj in enumerate(bpy.data.objects):
    if obj.type == 'CAMERA':
        scene.camera = obj
        print(f"Rendering from camera: {obj.name}")
        
        # Update file paths for outputs
        for output_type in output_types:
            outputs[output_type].file_slots[0].path = f"camera_{index:03d}_"

        # Render
        bpy.ops.render.render(write_still=True)

# Save the blend file
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_dir, "scene.blend"))
