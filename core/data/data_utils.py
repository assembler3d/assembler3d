import os
import numpy as np
import open3d as o3d

def load_txt(list_file):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def random_rotation_matrix():
    """
    Create a random rotation matrix.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    R = np.array([[cos_angle + axis[0]**2 * (1 - cos_angle),
                   axis[0]*axis[1] * (1 - cos_angle) - axis[2]*sin_angle,
                   axis[0]*axis[2] * (1 - cos_angle) + axis[1]*sin_angle],
                  [axis[1]*axis[0] * (1 - cos_angle) + axis[2]*sin_angle,
                   cos_angle + axis[1]**2 * (1 - cos_angle),
                   axis[1]*axis[2] * (1 - cos_angle) - axis[0]*sin_angle],
                  [axis[2]*axis[0] * (1 - cos_angle) - axis[1]*sin_angle,
                   axis[2]*axis[1] * (1 - cos_angle) + axis[0]*sin_angle,
                   cos_angle + axis[2]**2 * (1 - cos_angle)]])
    return R

def random_mirror_matrix():
    """
    Create a random mirror matrix.
    """
    if np.random.rand() < 0.75:
        axis = np.random.choice([0, 1, 2], size=1)[0]
        M = np.eye(3)
        M[axis, axis] = -1
    else:
        M = np.eye(3)
    return M


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

def transform(points, normals):
    points = points - np.mean(points, axis=0)
    # random rotation
    R = random_rotation_matrix()
    transformed_points, transformed_normals = apply_transformation(points, normals, R)
    return transformed_points, transformed_normals


def save_color_ply(filepath, vertices, colors):
    points = vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filepath, point_cloud,write_ascii=True)
