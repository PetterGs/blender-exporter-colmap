import numpy as np
import math
import bpy
from bpy.types import Object
from mathutils import Vector
import random

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

def screen_to_camera_plane(x: float, y: float, z_depth: float, cam: Object, width: int, height: int) -> tuple[float, float, float]:
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

def is_in_frustum(point_3d: np.ndarray, camera: bpy.types.Object, width: int, height: int) -> bool:
    """
    Determines if a point is within the camera's frustum.
    
    :param point_3d: The 3D point in world coordinates.
    :param camera: The camera object to check the frustum against.
    :return: True if the point is within the camera's frustum, False otherwise.
    """
    # Convert the point to a Vector (Blender uses Vector math for transformations)
    point_world = Vector(point_3d)

    # Step 1: Transform the point into the camera's local space
    point_camera = camera.matrix_world.inverted() @ point_world

    # Step 2: Get the camera parameters
    near_clip = camera.data.clip_start
    far_clip = camera.data.clip_end
    fov = camera.data.angle  # Full horizontal FOV in radians
    aspect_ratio = width / height

    # Step 3: Check if the point is between near and far clip planes
    if point_camera.z > -near_clip or point_camera.z < -far_clip:
        return False  # Point is outside near/far planes

    # Step 4: Calculate the horizontal and vertical boundaries at the point's depth
    tan_half_fov = math.tan(fov / 2)
    near_height = tan_half_fov * abs(point_camera.z)  # Height of the frustum at the point's z-depth
    near_width = near_height * aspect_ratio  # Width is scaled by the aspect ratio

    # Step 5: Check the point against the frustum bounds
    if abs(point_camera.x) > near_width:
        return False  # Point is outside left/right planes
    if abs(point_camera.y) > near_height:
        return False  # Point is outside top/bottom planes

    # If all checks pass, the point is inside the frustum
    return True

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

def create_emission_material(mesh_object: Object) -> None:
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
    def __init__(self, depth_images: list[str], color_images: list[str], cameras: list[Object], grid_resolution: int = 128, distance_threshold: float = 0.01):
        """
        Initializes the UnprunedPoints object by accumulating points from the provided
        depth images, color images, and cameras, while performing the merge after each accumulation.
        
        :param depth_images: List of file paths to depth images (EXR format)
        :param color_images: List of file paths to color images (JPG format)
        :param cameras: List of Blender camera objects corresponding to the images
        :param grid_resolution: Grid resolution for merging points
        :param distance_threshold: Distance threshold for merging nearby points
        """
        self.vertices = []
        """The locations of the points in 3D space."""
        self.colors = []
        """The colors of the points."""
        self.frustum_cameras = []
        """The indices of the cameras of which the point is visible withing the frustum."""

        self.grid_resolution = grid_resolution
        """The grid resolution for merging points."""
        self.distance_threshold = distance_threshold
        """The distance threshold for merging nearby points."""
        self.accumulated_images_total = 0
        """The total number of accumulated images so far"""

        # Ensure the number of depth images, color images, and cameras match
        if len(depth_images) != len(color_images) or len(depth_images) != len(cameras):
            raise ValueError("Mismatch in number of depth images, color images, and cameras.")

        # Accumulate and merge points from each depth and color image using the provided cameras
        for depth_image_path, color_image_path, cam in zip(depth_images, color_images, cameras):
            self.accumulate_from_image(depth_image_path, color_image_path, cam, cameras)

            if self.accumulated_images_total < 2:
                continue

            self.randomize()
            self.merge_nearby_points()

    def add_point(self, vertex: np.ndarray, color: np.ndarray, frustum_cameras: list[int]):
        """Adds a vertex, its corresponding color, and the list of frustum cameras."""
        self.vertices.append(vertex)
        self.colors.append(color)
        self.frustum_cameras.append(frustum_cameras)

    def accumulate_from_image(self, depth_map_path: str, rgb_map_path: str, camera: Object, cameras: list[Object]):
        """Accumulates points and colors from a depth and RGB image projected from the camera into shared lists."""
        depth_array, rgb_array, render_width, render_height = load_depth_and_rgb(depth_map_path, rgb_map_path)

        for y in range(render_height):
            for x in range(render_width):
                z_depth = depth_array[y, x]
                if z_depth == 0 or z_depth >= camera.data.clip_end:
                    continue

                point_3d = screen_to_camera_plane(x, y, z_depth, camera, render_width, render_height)
                rgb = rgb_array[y, x]

                # Check which cameras' frustums contain this point
                frustum_cameras = []
                for index, cam in enumerate(cameras):
                    if cam == camera:
                        frustum_cameras.append(index)
                        continue
                    if is_in_frustum(point_3d, cam, render_width, render_height):
                        frustum_cameras.append(index)

                self.add_point(point_3d, rgb, frustum_cameras)

        self.accumulated_images_total += 1

    def randomize(self):
        """Randomizes the order of vertices, colors, and frustum cameras."""
        combined = list(zip(self.vertices, self.colors, self.frustum_cameras))
        random.shuffle(combined)
        self.vertices, self.colors, self.frustum_cameras = zip(*combined)
        self.vertices = list(self.vertices)
        self.colors = list(self.colors)
        self.frustum_cameras = list(self.frustum_cameras)

    def merge_nearby_points(self):
        """Merges nearby points using grid-based spatial hashing and combines frustum_cameras for merged points."""
        vertices = np.array(self.vertices)
        colors = np.array(self.colors)
        frustum_cameras = self.frustum_cameras  # No need to convert to a numpy array
        
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)
        box_size = max_bound - min_bound
        cell_size = box_size / self.grid_resolution
        grid = {}

        for i, vertex in enumerate(vertices):
            cell = tuple(((vertex - min_bound) // cell_size).astype(int))
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)

        to_delete = set()
        new_vertices = []
        new_colors = []
        new_frustum_cameras = []

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
            combined_frustum_cameras = set(frustum_cameras[i])  # Combine frustum cameras as a set to avoid duplicates

            nearby_points = [i]
            nearby_color_sum = mean_color.copy()

            for nearby_cell in get_nearby_cells(cell):
                if nearby_cell not in grid:
                    continue
                for j in grid[nearby_cell]:
                    if j == i or j in to_delete:
                        continue
                    if np.linalg.norm(vertices[i] - vertices[j]) <= self.distance_threshold:
                        nearby_points.append(j)
                        mean_location += vertices[j]
                        nearby_color_sum += colors[j]
                        combined_frustum_cameras.update(frustum_cameras[j])  # Combine frustum cameras
                        to_delete.add(j)

            if len(nearby_points) > 1:
                mean_location /= len(nearby_points)
                mean_color = nearby_color_sum / len(nearby_points)

            new_vertices.append(mean_location)
            new_colors.append(mean_color)
            new_frustum_cameras.append(list(combined_frustum_cameras))  # Store merged frustum cameras

        self.vertices = new_vertices
        self.colors = new_colors
        self.frustum_cameras = new_frustum_cameras

    def create_point_cloud(self) -> Object:
        """Converts the stored vertices and colors into a point cloud."""

        point_cloud = self.create_point_cloud_object(self.vertices, self.colors, self.frustum_cameras)

        emission_mat = create_emission_material(point_cloud)

        add_geometry_nodes(point_cloud, emission_mat)
        
        return point_cloud
    
    def create_point_cloud_object(self, vertices: list[tuple[float, float, float]], colors: list[tuple[float, float, float]]) -> Object:
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
    
    def create_point_cloud_object(self, vertices: list[tuple[float, float, float]], colors: list[tuple[float, float, float]], frustum_cameras: list[list[int]]) -> Object:
        """Creates a point cloud mesh from vertices and assigns colors using color attributes."""
        
        # Create a new mesh and object
        mesh_data = bpy.data.meshes.new("PointCloud")
        mesh_object = bpy.data.objects.new("PointCloud", mesh_data)
        bpy.context.scene.collection.objects.link(mesh_object)

        allowed_camera_index = 2

        filtered_vertices = []
        filtered_colors = []

        for i, vertex in enumerate(vertices):
            camera_indices = frustum_cameras[i]
            if allowed_camera_index in camera_indices:
                filtered_vertices.append(vertices[i])
                filtered_colors.append(colors[i])

        # Add vertices to the mesh (no faces for point cloud)
        mesh_data.from_pydata(filtered_vertices, [], [])
        mesh_data.update()

        color_layer = mesh_data.color_attributes.new(name="Col", type='BYTE_COLOR', domain='POINT')

        # Assign colors to the vertices (which are points in this case)
        for i, vertex in enumerate(mesh_data.vertices):
            r, g, b = filtered_colors[i]  # Get the color for the current vertex
            color_layer.data[i].color = (r, g, b, 1.0)
            
        return mesh_object

