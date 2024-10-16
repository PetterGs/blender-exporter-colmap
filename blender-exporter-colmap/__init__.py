import os
import numpy as np
from pathlib import Path
import math
import bpy
import mathutils
from mathutils import Vector
from . ext.read_write_model import write_model, Camera, Image
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty
from bpy.types import CompositorNodeOutputFile
import random

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

class UnprunedPoints:
    def __init__(self, depth_images: list[str], color_images: list[str], cameras: list[bpy.types.Object]):
        """
        Initializes the UnprunedPoints object by accumulating points from the provided
        depth images, color images, and cameras.
        
        :param depth_images: List of file paths to depth images (EXR format)
        :param color_images: List of file paths to color images (JPG format)
        :param cameras: List of Blender camera objects corresponding to the images
        """
        self.vertices = []
        self.colors = []

        # Ensure the number of depth images, color images, and cameras match
        if len(depth_images) != len(color_images) or len(depth_images) != len(cameras):
            raise ValueError("Mismatch in number of depth images, color images, and cameras.")

        # Accumulate points from the depth and color images using the provided cameras
        for depth_image_path, color_image_path, cam in zip(depth_images, color_images, cameras):
            self.accumulate_from_image(depth_image_path, color_image_path, cam)

    def add_point(self, vertex: np.ndarray, color: np.ndarray):
        """Adds a vertex and its corresponding color."""
        self.vertices.append(vertex)
        self.colors.append(color)

    def accumulate_from_image(self, depth_map_path: str, rgb_map_path: str, camera: bpy.types.Object):
        """Accumulates points and colors from a depth and RGB image projected from the camera into shared lists."""
        depth_array, rgb_array, render_width, render_height = load_depth_and_rgb(depth_map_path, rgb_map_path)

        for y in range(render_height):
            for x in range(render_width):
                z_depth = depth_array[y, x]
                if z_depth == 0 or z_depth >= camera.data.clip_end:
                    continue

                point_3d = screen_to_camera_plane(x, y, z_depth, camera, render_width, render_height)
                rgb = rgb_array[y, x]
                self.add_point(point_3d, rgb)

    def randomize(self):
        """Randomizes the order of vertices and their corresponding colors."""
        combined = list(zip(self.vertices, self.colors))
        random.shuffle(combined)
        self.vertices, self.colors = zip(*combined)
        self.vertices = list(self.vertices)
        self.colors = list(self.colors)

    def merge_nearby_points(self, grid_resolution: int = 128, distance_threshold: float = 0.01):
        """Merges nearby points using grid-based spatial hashing."""
        vertices = np.array(self.vertices)
        colors = np.array(self.colors)
        
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)
        box_size = max_bound - min_bound
        cell_size = box_size / grid_resolution
        grid = {}

        for i, vertex in enumerate(vertices):
            cell = tuple(((vertex - min_bound) // cell_size).astype(int))
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)

        to_delete = set()
        new_vertices = []
        new_colors = []

        def get_nearby_cells(cell):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        yield (cell[0] + dx, cell[1] + dy, cell[2] + dz)

        for i, vertex in enumerate(vertices):
            if i in to_delete:
                continue

            cell = tuple(((vertex - min_bound) // cell_size).astype(int))
            mean_location = vertex.copy()
            mean_color = colors[i].copy()

            nearby_points = [i]
            nearby_color_sum = mean_color.copy()

            for nearby_cell in get_nearby_cells(cell):
                if nearby_cell not in grid:
                    continue
                for j in grid[nearby_cell]:
                    if j == i or j in to_delete:
                        continue
                    if np.linalg.norm(vertices[i] - vertices[j]) <= distance_threshold:
                        nearby_points.append(j)
                        mean_location += vertices[j]
                        nearby_color_sum += colors[j]
                        to_delete.add(j)

            if len(nearby_points) > 1:
                mean_location /= len(nearby_points)
                mean_color = nearby_color_sum / len(nearby_points)

            new_vertices.append(mean_location)
            new_colors.append(mean_color)

        self.vertices = new_vertices
        self.colors = new_colors

    def create_point_cloud(self) -> bpy.types.Object:
        """Converts the stored vertices and colors into a point cloud."""


        point_cloud = self.create_point_cloud_object(self.vertices, self.colors)

        emission_mat = create_emission_material(point_cloud)

        add_geometry_nodes(point_cloud, emission_mat)
        
        return point_cloud

    def create_point_cloud_object(self, vertices: list[tuple[float, float, float]], colors: list[tuple[float, float, float]]) -> bpy.types.Object:
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

        # List of depth image paths and color image paths
        depth_images = [next((i for i in temp_exr_files if f"{cam.name_full}_depth" in i), None) for cam in sorted_cameras]
        color_images = [os.path.join(images_dir, f'{cam.name_full}.jpg') for cam in sorted_cameras]

        # Remove None entries from depth_images and color_images
        depth_images = [img for img in depth_images if img is not None]
        color_images = [color_images[i] for i in range(len(depth_images))]  # Ensure lists stay in sync

        # Initialize UnprunedPoints with depth and color images and corresponding cameras
        unpruned_points = UnprunedPoints(depth_images, color_images, sorted_cameras)

        # Randomize point order
        unpruned_points.randomize()

        # Merge nearby points
        unpruned_points.merge_nearby_points()

        unpruned_points.create_point_cloud()
        
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
