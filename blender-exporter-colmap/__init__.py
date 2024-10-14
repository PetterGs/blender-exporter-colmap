import os
import numpy as np
from pathlib import Path
import math
import bpy
import mathutils
from mathutils import Vector
from . ext.read_write_model import write_model, Camera, Image
from bpy_extras.io_utils import ExportHelper
from mathutils.bvhtree import BVHTree
from bpy.props import StringProperty
from bpy.types import CompositorNodeOutputFile
import random

import bmesh
from bmesh.types import BMEdge, BMesh

bl_info = {
    "name": "Scene exporter for colmap",
    "description": "Generates a dataset for colmap by exporting blender camera poses and rendering scene.",
    "author": "Ohayoyogi",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "File/Export",
    "warning": "",
    "wiki_url": "https://github.com/ohayoyogi/blender-exporter-colmap",
    "tracker_url": "https://github.com/ohayoyogi/blender-exporter-colmap/issues",
    "category": "Import-Export"
}


def load_depth_and_rgb(depth_map_path: str, rgb_map_path: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Loads depth and RGB data from EXR files."""
    depth_image = bpy.data.images.load(depth_map_path)
    rgb_image = bpy.data.images.load(rgb_map_path)

    # Get image dimensions
    render_width = depth_image.size[0]
    render_height = depth_image.size[1]

    # Extract depth and RGB pixel values
    depth_pixels = np.array(depth_image.pixels[:]).reshape(render_height, render_width, 4)
    rgb_pixels = np.array(rgb_image.pixels[:]).reshape(render_height, render_width, 4)

    depth_array = depth_pixels[:, :, 0]  # R channel contains the depth
    rgb_array = rgb_pixels[:, :, :3]     # R, G, B channels for colors

    return depth_array, rgb_array, render_width, render_height

def screen_to_camera_plane(x: float, y: float, z_depth: float, cam: bpy.types.Object, width: int, height: int) -> tuple[float, float, float]:
    """Projects 2D screen coordinates onto the camera's projection plane in 3D space, using depth values."""

    # Convert 2D screen coordinates to normalized device coordinates (NDC) [-1, 1]
    ndc_x = (x / width) * 2 - 1
    ndc_y = (y / height) * 2 - 1

    # Calculate how far the points are spread in X and Y based on the half-FOV
    half_fov = cam.data.angle / 2
    tan_half_fov = math.tan(half_fov)

    # Calculate camera space coordinates for X and Y using the FOV, aspect ratio, and depth
    aspect_ratio = width / height
    camera_x = ndc_x * z_depth * tan_half_fov  # Scale by z_depth to account for distance
    camera_y = ndc_y * z_depth * tan_half_fov / aspect_ratio  # Y is scaled by the aspect ratio

    # Z-coordinate is based on the depth value
    camera_z = -z_depth  # Depth is along the negative Z-axis in camera space

    # Transform from camera space to world space
    camera_coords = Vector((camera_x, camera_y, camera_z))
    world_coords = cam.matrix_world @ camera_coords

    return world_coords

def create_point_cloud(vertices: list[tuple[float, float, float]], colors: list[tuple[float, float, float]]) -> bpy.types.Object:
    """Creates a point cloud mesh from vertices and assigns colors using color attributes."""
    
    # Create a new mesh and object
    mesh_data = bpy.data.meshes.new("PointCloud")
    mesh_object = bpy.data.objects.new("PointCloud", mesh_data)
    bpy.context.scene.collection.objects.link(mesh_object)

    # Add vertices to the mesh (no faces for point cloud)
    mesh_data.from_pydata(vertices, [], [])
    mesh_data.update()

    color_layer = mesh_data.color_attributes.new(name="Col", type='BYTE_COLOR', domain='POINT')

    # Assign colors to the vertices (which are points in this case)
    for i, vertex in enumerate(mesh_data.vertices):
        r, g, b = colors[i]  # Get the color for the current vertex
        color_layer.data[i].color = (r, g, b, 1.0)
        
    return mesh_object

def add_geometry_nodes(mesh_object, material):

    bpy.ops.object.select_all(action='DESELECT')

    #select the object
    bpy.context.view_layer.objects.active = mesh_object
    mesh_object.select_set(True)

    bpy.ops.node.new_geometry_nodes_modifier()
    
    node_tree = bpy.data.node_groups["Geometry Nodes"]
    
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = node_tree.nodes["Group Input"]
    output_node = node_tree.nodes["Group Output"]

    mesh_to_points = nodes.new(type='GeometryNodeMeshToPoints')
    mesh_to_points.inputs['Radius'].default_value = 0.005

    set_material_node = nodes.new(type='GeometryNodeSetMaterial')
    set_material_node.inputs['Material'].default_value = material

    links.new(input_node.outputs[0], mesh_to_points.inputs['Mesh'])
    links.new(mesh_to_points.outputs['Points'], set_material_node.inputs['Geometry'])
    links.new(set_material_node.outputs['Geometry'], output_node.inputs[0])
    
    print("Geometry Nodes setup completed.")

def create_emission_material(mesh_object: bpy.types.Object) -> None:
    """Creates an unlit emission shader for the point cloud."""
    mat = bpy.data.materials.new(name="PointCloudMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create an emission shader
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    emission_node = nodes.new(type="ShaderNodeEmission")
    color_attr_node = nodes.new(type="ShaderNodeVertexColor")

    # Link the shader nodes
    links.new(color_attr_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    # Assign the vertex color attribute node to the "Col" layer
    color_attr_node.layer_name = "Col"

    # Assign the material to the object
    if mesh_object.data.materials:
        mesh_object.data.materials[0] = mat
    else:
        mesh_object.data.materials.append(mat)

    return mat

def accumulate_points_from_image(depth_map_path: str, rgb_map_path: str, camera: bpy.types.Object, vertices: list, colors: list) -> None:
    """Accumulates points and colors from a depth and RGB image projected from the camera into shared lists."""
    
    # Load depth and RGB data
    depth_array, rgb_array, render_width, render_height = load_depth_and_rgb(depth_map_path, rgb_map_path)

    # Iterate over every pixel and project to the camera plane
    for y in range(render_height):
        for x in range(render_width):
            # Get the depth for the current pixel
            z_depth = depth_array[y, x]

            # Skip pixels with no valid depth data
            if z_depth == 0 or z_depth >= camera.data.clip_end:
                continue

            # Project the pixel to 3D coordinates on the camera's projection plane
            point_3d = screen_to_camera_plane(x, y, z_depth, camera, render_width, render_height)
            vertices.append(point_3d)

            # Get the RGB values for this pixel and store them
            rgb = rgb_array[y, x]
            colors.append(rgb)

def get_bounding_box(mesh_object):
    """Returns the min and max coordinates of the mesh's bounding box."""
    vertices = [v.co for v in mesh_object.data.vertices]
    min_bound = Vector((min(v.x for v in vertices), 
                        min(v.y for v in vertices), 
                        min(v.z for v in vertices)))
    max_bound = Vector((max(v.x for v in vertices), 
                        max(v.y for v in vertices), 
                        max(v.z for v in vertices)))
    return min_bound, max_bound

def hash_point(point, min_bound, grid_resolution, cell_size):
    """Converts a 3D point into a grid cell coordinate based on the bounding box and grid resolution."""
    relative_pos = point - min_bound  # Shift point relative to bounding box min corner
    return (int(relative_pos.x // cell_size.x),
            int(relative_pos.y // cell_size.y),
            int(relative_pos.z // cell_size.z))

def merge_point_cloud_nearby_points(mesh_object: bpy.types.Object, grid_resolution: int = 128, distance_threshold: float = 0.01) -> None:
    """Updates the existing point cloud mesh with merged vertices and colors using grid-based spatial hashing."""
    
    bm = bmesh.new()
    bm.from_mesh(mesh_object.data)

    # Ensure the vertex lookup table is built
    bm.verts.ensure_lookup_table()

    color_layer = bm.loops.layers.color.get("Col")

    if len(bm.verts) == 0:
        raise ValueError("No vertices found in the mesh!")

    print(f"Merging point cloud with {len(bm.verts)} vertices.")

    # Get bounding box of the mesh
    min_bound, max_bound = get_bounding_box(mesh_object)

    # Calculate the total size of the bounding box in each dimension
    box_size = max_bound - min_bound

    # Calculate cell size based on grid resolution
    cell_size = Vector((box_size.x / grid_resolution,
                        box_size.y / grid_resolution,
                        box_size.z / grid_resolution))

    # Create the spatial hash grid
    grid = {}

    # Populate the grid with vertices
    for vert in bm.verts:
        cell = hash_point(vert.co, min_bound, grid_resolution, cell_size)
        if cell not in grid:
            grid[cell] = []
        grid[cell].append(vert)

    # Store vertices to delete later
    to_delete_verts = set()

    # Function to get neighboring grid cells (3x3x3 surrounding cells)
    def get_nearby_cells(cell):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    yield (cell[0] + dx, cell[1] + dy, cell[2] + dz)

    # Loop over all vertices and compare to find nearby ones
    for vert in bm.verts:
        if vert in to_delete_verts:
            continue  # Skip vertices marked for deletion

        # Get the cell the vertex is in
        cell = hash_point(vert.co, min_bound, grid_resolution, cell_size)

        mean_location = vert.co.copy()
        mean_color = Vector((0, 0, 0, 0))

        nearby_vertices = [vert]  # Include the current vertex itself

        # Accumulate the color of the current vertex
        for loop in vert.link_loops:
            mean_color += Vector(loop[color_layer])

        # Look for nearby vertices in the surrounding cells
        for nearby_cell in get_nearby_cells(cell):
            if nearby_cell not in grid:
                continue
            for other_vert in grid[nearby_cell]:
                if other_vert == vert or other_vert in to_delete_verts:
                    continue

                # Check if the vertex is within the distance threshold
                if (vert.co - other_vert.co).length <= distance_threshold:
                    nearby_vertices.append(other_vert)
                    mean_location += other_vert.co

                    # Accumulate color
                    for loop in other_vert.link_loops:
                        mean_color += Vector(loop[color_layer])

                    # Mark the other vertex for deletion
                    to_delete_verts.add(other_vert)

        # Only merge if there are valid nearby vertices
        if len(nearby_vertices) > 1:
            # Compute the average position and color
            new_location = mean_location / len(nearby_vertices)
            new_color = mean_color / len(nearby_vertices)

            # Update the main vertex to the averaged location and color
            vert.co = new_location

            for loop in vert.link_loops:
                loop[color_layer] = new_color

    # Remove all vertices marked for deletion
    bmesh.ops.delete(bm, geom=list(to_delete_verts), context='VERTS')

    bm.verts.ensure_lookup_table()
    verts_total = len(bm.verts)
    # Write the updated BMesh back to the original mesh
    bm.to_mesh(mesh_object.data)
    mesh_object.data.update()

    bm.free()

    print(f"Point cloud merged with {verts_total} vertices remaining.")

def randomize_order(vertices: list, colors: list) -> tuple:
    """Randomizes the order of vertices and their corresponding colors."""
    combined = list(zip(vertices, colors))  # Combine vertices and colors to keep their correspondence
    random.shuffle(combined)  # Shuffle the combined list
    vertices, colors = zip(*combined)  # Unzip back into separate lists
    return list(vertices), list(colors)

def create_point_cloud_after_accumulation(vertices: list, colors: list) -> None:
    """Creates the point cloud mesh after all points have been accumulated."""
    
    vertices, colors = randomize_order(vertices, colors)
    
    mesh_object = create_point_cloud(vertices, colors)

    merge_point_cloud_nearby_points(mesh_object)

    emission_mat = create_emission_material(mesh_object)

    add_geometry_nodes(mesh_object, emission_mat)

class BlenderExporterForColmap(bpy.types.Operator, ExportHelper):

    filename_ext = "."

    directory: StringProperty()

    filter_folder = True

    def setup_depth_rendering(self) -> CompositorNodeOutputFile:
        # Ensure Z pass is enabled
        bpy.context.view_layer.use_pass_z = True

        # Set the scene to use nodes (compositor)
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree

        # Clear any existing nodes
        for node in tree.nodes:
            tree.nodes.remove(node)

        # Create the render layer node
        render_layers_node = tree.nodes.new(type="CompositorNodeRLayers")

        # Create the file output node for the depth map
        file_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        file_output_node.format.file_format = 'OPEN_EXR' 
        file_output_node.format.color_depth = '32'

        # Link the Z output from the Render Layers node to the File Output node
        tree.links.new(render_layers_node.outputs["Depth"], file_output_node.inputs[0])

        return file_output_node
    
    def export_dataset(self, context, dirpath: Path, format: str):
        scene = context.scene
        scene_cameras = [i for i in scene.objects if i.type == "CAMERA"]

        output_format = format if format in ['.txt', '.bin'] else '.txt'

        scale = scene.render.resolution_percentage / 100.0

        output_dir = dirpath
        images_dir = output_dir / 'images'

        output_dir.mkdir(parents=True, exist_ok=True)

        cameras = {}
        images = {}

        depth_file_output_node = self.setup_depth_rendering()
        depth_file_output_node.base_path = bpy.app.tempdir
        sorted_cameras = sorted(scene_cameras, key=lambda x: x.name_full + ".jpg")
        for idx, cam in enumerate(sorted_cameras):
            camera_id = idx+1
            filename = f'{cam.name_full}.jpg'
            depth_file_output_node.file_slots[0].path = f'{cam.name_full}_depth'
            width = scene.render.resolution_x
            height = scene.render.resolution_y
            focal_length = cam.data.lens
            sensor_width = cam.data.sensor_width
            sensor_height = cam.data.sensor_height
            fx = focal_length * width / sensor_width
            fy = focal_length * height / sensor_height
            # fx, fy, cx, cy, k1, k2, p1, p2
            params = [fx, fy, width/2, height/2, 0, 0, 0, 0]
            cameras[camera_id] = Camera(
                id=camera_id,
                model='OPENCV',
                width=width,
                height=height,
                params=params
            )

            image_id = camera_id
            rotation_mode_bk = cam.rotation_mode

            cam.rotation_mode = "QUATERNION"
            cam_rot_orig = mathutils.Quaternion(cam.rotation_quaternion)
            cam_rot = mathutils.Quaternion((
                cam_rot_orig.x,
                cam_rot_orig.w,
                cam_rot_orig.z,
                -cam_rot_orig.y))
            qw = cam_rot.w
            qx = cam_rot.x
            qy = cam_rot.y
            qz = cam_rot.z
            cam.rotation_mode = rotation_mode_bk

            T = mathutils.Vector(cam.location)
            T1 = -(cam_rot.to_matrix() @ T)

            tx = T1[0]
            ty = T1[1]
            tz = T1[2]
            images[image_id] = Image(
                id=image_id,
                qvec=np.array([qw, qx, qy, qz]),
                tvec=np.array([tx, ty, tz]),
                camera_id=camera_id,
                name=filename,
                xys=[],
                point3D_ids=[]
            )

            # Render scene
            image_path = os.path.join(images_dir, filename) 
            bpy.context.scene.camera = cam
            bpy.ops.render.render()
            bpy.data.images['Render Result'].save_render(image_path)

            yield 100.0 * idx / (len(scene_cameras) + 1)

        # TODO: Figure out a way to add direct depth map filenames instead of relying on Blenders clunky naming system.
        temp_exr_files = []
        for root, dirs, files in os.walk(depth_file_output_node.base_path):
            for file in files:
                if file.endswith(".exr"):
                    full_path = os.path.join(root, file)
                    temp_exr_files.append(full_path)

        # Initialize shared lists for vertices and colors
        vertices = []
        colors = []

        for idx, cam in enumerate(sorted_cameras):
            camera_id = idx+1
            depth_map_file = next((i for i in temp_exr_files if f"{cam.name_full}_depth" in i), None)
            rgb_map_file = os.path.join(images_dir, f'{cam.name_full}.jpg') 
            if depth_map_file is None:
                continue  

            # Accumulate points and colors from the current image
            accumulate_points_from_image(depth_map_file, rgb_map_file, cam, vertices, colors)

        create_point_cloud_after_accumulation(vertices, colors)
        
        point_cloud_path = os.path.join(output_dir, 'points3D.ply')
        bpy.ops.wm.ply_export(filepath=point_cloud_path, export_selected_objects=True, apply_modifiers=False)

        write_model(cameras, images, {}, str(output_dir), output_format)
        yield 100.0

    def execute_(self, context, format):
        dirpath = Path(self.directory)
        if not dirpath.is_dir():
            return {"WARNING", "Illegal directory was passed: " + self.directory}

        context.window_manager.progress_begin(0, 100)
        for progress in self.export_dataset(context, dirpath, format):
            context.window_manager.progress_update(progress)
        context.window_manager.progress_end()

        return {"FINISHED"}


class BlenderExporterForColmapBinary(BlenderExporterForColmap):
    bl_idname = "object.colmap_dataset_generator_binary"
    bl_label = "Export as colmap dataset with binary format"
    bl_options = {"PRESET"}

    def execute(self, context):
        return super().execute_(context, '.bin')


class BlenderExporterForColmapText(BlenderExporterForColmap):
    bl_idname = "object.colmap_dataset_generator_text"
    bl_label = "Export as colmap dataset with text format"
    bl_options = {"PRESET"}

    def execute(self, context):
        return super().execute_(context, '.txt')


def _blender_export_operator_function(topbar_file_import, context):
    topbar_file_import.layout.operator(
        BlenderExporterForColmapText.bl_idname, text="Colmap dataset (.txt)"
    )
    topbar_file_import.layout.operator(
        BlenderExporterForColmapBinary.bl_idname, text="Colmap dataset (.bin)"
    )


def register():
    bpy.utils.register_class(BlenderExporterForColmapBinary)
    bpy.utils.register_class(BlenderExporterForColmapText)
    bpy.types.TOPBAR_MT_file_export.append(_blender_export_operator_function)


def unregister():
    bpy.utils.unregister_class(BlenderExporterForColmapBinary)
    bpy.utils.unregister_class(BlenderExporterForColmapText)


if __name__ == "__main__":
    register()
