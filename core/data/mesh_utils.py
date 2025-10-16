import trimesh
import numpy as np
import networkx as nx
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import copy

def save_color_ply(filepath, vertices, colors):
    points = vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filepath, point_cloud,write_ascii=True)


def load_mesh(mesh_path):
    """
    Load a mesh from path.
    
    Args:
        mesh_path: str, path to mesh file
    
    Returns:
        trimesh.Trimesh
    """
    mesh = trimesh.load(mesh_path, process=True, skip_materials=True, force='mesh')
    if isinstance(mesh, trimesh.path.Path3D):
        return None
    mesh = mesh.process(validate=True, merge_tex=True, merge_norm=True)
    return mesh


def return_normalization_scale_center(mesh):
    """Return the scale of the mesh
    """
    # Get mesh bounds
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max() / 2
    return scale, center

def scale_part_mesh(mesh, scale, center):
    # Scale vertices
    vertices = (mesh.vertices - center) / scale
    
    # Create new normalized mesh
    normalized_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=mesh.faces,
    )
    return normalized_mesh

def normalize_mesh(mesh, bound=1.0):
    """
    Normalize mesh to fit in a cube with specified bounds.
    
    Args:
        mesh: trimesh.Trimesh object
        bound: float, the target boundary size (default: 1.0)
               mesh will be scaled to fit in a cube of [-bound, bound]
    Returns:
        trimesh.Trimesh: normalized mesh
    """
    # Get mesh bounds
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max() / 2
    
    # Normalize vertices
    vertices = (mesh.vertices - center) / scale * bound
    
    # Create new normalized mesh
    normalized_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=mesh.faces,
    )
        
    return normalized_mesh

def rotate_mesh(mesh, rotation):
    """
    Apply a rotation transformation to the mesh.
    
    Args:
        mesh: trimesh.Trimesh object
        rotation: np.array of shape (3,3)
    Returns:
        trimesh.Trimesh: transformed mesh
    """
    mesh.vertices = np.dot(mesh.vertices, rotation.T)
    return mesh

def center_mesh(mesh):
    """
    Center a mesh at the origin.
    
    Args:
        mesh: trimesh.Trimesh object
    Returns:
        trimesh.Trimesh: transformed mesh
    """
    # note that center_mass not works for non-watertight meshes
    mean = np.mean(mesh.vertices, axis=0)
    mesh.vertices -= mean
    return mesh

def compute_chamfer_distance(pc1, pc2):
    """
    Compute the Chamfer distance between two point clouds.

    Parameters:
    pc1 (numpy array): First point cloud (N x 3)
    pc2 (numpy array): Second point cloud (M x 3)

    Returns:
    float: Chamfer distance
    """
    # Create KD-trees for efficient nearest neighbor search
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(pc1)
    min_y_to_x = x_nn.kneighbors(pc2)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(pc2)
    min_x_to_y = y_nn.kneighbors(pc1)[0]
    chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    return chamfer_dist

def compute_transformation(P1, P2):
    """
    Compute a transformation matrix (rotation + translation) to transform one set of points to another, 
    using the least square fitting.
    
    Args:
        src_points: np.array of shape (N,3)
        tgt_points: np.array of shape (N,3)
    Returns:
        np.array of shape (4,4)
    """
    # Compute centroids
    centroid1 = np.mean(P1, axis=0)
    centroid2 = np.mean(P2, axis=0)
    
    # Center the points
    P1_centered = P1 - centroid1
    P2_centered = P2 - centroid2
    
    # Compute covariance matrix H
    H = P1_centered.T @ P2_centered
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = Vt.T @ U.T
    
    # Check for reflection to ensure right-hand coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation vector
    t = centroid2 - R @ centroid1
    
    # Create homogeneous transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    return transformation

def apply_transformation(points, normals, transform):
    """
    Apply a transformation matrix to points and normals.
    """
    transformed_points = np.dot(points, transform.T)
    if normals is not  None:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        normals /= norms
        
        transformed_normals = np.dot(normals, transform.T)
        norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        transformed_normals /= norms
    else:
        transformed_normals=None
    return transformed_points, transformed_normals

def bbox(points):
    from sklearn.decomposition import PCA
    try: 
        to_origin, size = trimesh.bounds.oriented_bounds(obj=points, angle_digits=1)
        center = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])
    
        xdir = to_origin[0, :3]
        ydir = to_origin[1, :3]
        zdir = to_origin[2, :3]
    except:
       points = np.array(points)
       pca = PCA()
       pca.fit(points)
       pcomps = pca.components_
       points_local = np.matmul(pcomps, points.transpose()).transpose()
       all_max = points_local.max(axis=0)
       all_min = points_local.min(axis=0)
       center = np.dot(np.linalg.inv(pcomps), (all_max + all_min) / 2)
       size = all_max - all_min
       xdir = pcomps[0, :]
       xdir /= np.linalg.norm(xdir)
       ydir = pcomps[1, :]
       ydir /= np.linalg.norm(ydir)
       zdir = np.cross(xdir, ydir)
       zdir /= np.linalg.norm(zdir)
    
    R = np.vstack([xdir, ydir, zdir]).transpose().astype(np.float32)
    t = center.astype(np.float32)
    return R, t, size

def canonicalize_points(points):
    R, t, size = bbox(points[:,:3])

    points_trans = np.dot(points[:,:3] - t, np.linalg.inv(R).transpose())
    _, normals_trans = apply_transformation(points[:,:3], points[:,3:], np.linalg.inv(R))
    points_trans = np.concatenate([points_trans, normals_trans], axis=1)
    return points_trans

def canonicalize_mesh(mesh):
    points = mesh.sample(1000)
    R, t, size = bbox(points)

    v = mesh.vertices
    part_v = np.dot(v -t, np.linalg.inv(R).transpose())

    part_mesh = trimesh.Trimesh(vertices=part_v, faces=mesh.faces)
    return part_mesh

def get_canonical_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R.T          # inverse rotation
    T[:3, 3] = -R.T @ t      # inverse translation
    return T

def get_uncanonical_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def hybrid_robust_transform(source, target, threshold=0.02):
    """Compute the transformation matrix between two point clouds, 
    using both least-squares and global registration.
    """
    # first, compute the direct transformation
    direct_transform = compute_transformation(source, target)
    # transform source using direct transform
    source_transformed = source @ direct_transform[:3, :3].T + direct_transform[:3, 3]
    # compute the chamfer distance between transformed source and target
    chamfer_distance = compute_chamfer_distance(source_transformed, target)
    # if the chamfer distance is too large, use global registration instead
    if chamfer_distance > threshold:
        #print("Using PCA-based canonical registration")
        # canonicalize the target
        R_t, t_t, size_t = bbox(target)
        target_canonical = np.dot(target - t_t, np.linalg.inv(R_t).transpose())

        R_s, t_s, size_s = bbox(source)
        source_canonical = np.dot(source - t_s, np.linalg.inv(R_s).transpose())
        
        canonical_distance = compute_chamfer_distance(source_canonical, target_canonical)
        # flip the source to check the symmetry issue, try 'x', 'y', 'z' axis to flip
        distances = []
        for axis in [0,1,2]:
            source_transformed = copy.deepcopy(source_canonical)
            source_transformed[:, axis] = -source_transformed[:, axis]
            new_chamfer_distance = compute_chamfer_distance(source_transformed, target_canonical)
            distances.append(new_chamfer_distance)
        # find the axis with the smallest chamfer distance
        if min(distances) < canonical_distance:
            min_index = np.argmin(distances)
            axis = [0,1,2][min_index]
            flip_transform = np.eye(4)
            flip_transform[axis, axis] = -1
        else:
            flip_transform = np.eye(4)
        
        source_transformed = source_canonical @ flip_transform[:3, :3].T + flip_transform[:3, 3]
        
        # compose two into a transformation matrix
        T_target = get_canonical_transform(R_t, t_t)     # world → target_canonical
        T_source = get_canonical_transform(R_s, t_s)     # world → source_canonical
        T_source_inv = get_uncanonical_transform(R_s, t_s)  # source_canonical → world
        T_target_inv = get_uncanonical_transform(R_t, t_t)  # target_canonical → world

        global_transform = T_target_inv @ flip_transform @ T_source

        # check the chamfer distance after global registration
        source_transformed = source @ global_transform[:3, :3].T + global_transform[:3, 3]
        new_chamfer_distance = compute_chamfer_distance(source_transformed, target)
        if new_chamfer_distance > chamfer_distance:
            #print("Global registration failed")
            return direct_transform
        else:
            #print(f"Global registration succeeded: {chamfer_distance}/{new_chamfer_distance}")
            return global_transform
    else:
        return direct_transform


def transform_mesh(mesh, transform):
    """
    Apply a transformation matrix to the mesh.
    
    Args:
        mesh: trimesh.Trimesh object
        transform: np.array of shape (4,4)
    Returns:
        trimesh.Trimesh: transformed mesh
    """
    new_vertices = np.dot(mesh.vertices, transform[:3, :3].T) + transform[:3, 3]
    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)

def transform_mesh_inplace(mesh, transform):
    """
    Apply a transformation matrix to the mesh in-place.
    
    Args:
        mesh: trimesh.Trimesh object
        transform: np.array of shape (4,4)
    Returns:
        None
    """
    mesh.vertices = np.dot(mesh.vertices, transform[:3, :3].T) + transform[:3, 3]
    return mesh


def split_mesh_by_vertex_connectivity(mesh: trimesh.Trimesh):
    """
    Split a mesh into connected components using trimesh's vertex adjacency graph.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        
    Returns:
        List[trimesh.Trimesh]: List of connected components
    """
    # Get vertex adjacency graph
    graph = trimesh.graph.vertex_adjacency_graph(mesh)
    
    # Get connected components using networkx
    components_vertex_indices = list(nx.connected_components(graph))
    
    # Split the mesh into components
    components = []
    for vertex_indices in components_vertex_indices:
        # Convert set to array for indexing
        vertex_indices = np.array(list(vertex_indices))
        
        # Create a mask for faces
        vertex_mask = np.zeros(len(mesh.vertices), dtype=bool)
        vertex_mask[vertex_indices] = True
        face_mask = vertex_mask[mesh.faces].all(axis=1)
        
        # Get faces for this component
        component_faces = mesh.faces[face_mask]
        
        # Create vertex index mapping
        vertex_idx_map = np.full(len(mesh.vertices), -1)
        vertex_idx_map[vertex_indices] = np.arange(len(vertex_indices))
        
        # Remap face indices
        component_faces = vertex_idx_map[component_faces]
        
        # Create new mesh
        component_vertices = mesh.vertices[vertex_indices]
        component_mesh = trimesh.Trimesh(
            vertices=component_vertices,
            faces=component_faces
        )
        components.append(component_mesh)
    
    return components


def visualize_patches(patches):
    """
    Visualize all patches with different colors.

    Args:
        patches: List[trimesh.Trimesh] patches to visualize
    
    Returns:
        trimesh.Scene: patches with different face colors
    """
    n_patches = len(patches)
    colors = np.random.rand(n_patches, 4)
    colors[:, 3] = 1.0
    
    scene = trimesh.Scene()
    for i, (patch, color) in enumerate(zip(patches, colors)):
        patch.visual.material = trimesh.visual.material.SimpleMaterial(
            name=f'material_{i}',
            diffuse=color[:3]
        )
        scene.add_geometry(patch)
    
    return scene, colors
