import open3d as o3d
import numpy as np
import os
from sklearn.metrics import pairwise_distances
import ot

import numpy as np
import open3d as o3d
from ot.gromov import gromov_wasserstein

def compute_gromov_wasserstein(cloud1, cloud2):
    """
    Compute Gromov-Wasserstein distance between two point clouds using the POT library.
    """
    # Convert Open3D clouds to numpy arrays
    points1 = np.asarray(cloud1.points)
    points2 = np.asarray(cloud2.points)

    # Number of points in each point cloud
    n1 = points1.shape[0]
    n2 = points2.shape[0]

    # Compute pairwise Euclidean distances between points in cloud1 and cloud2
    dist_matrix_1 = np.linalg.norm(points1[:, np.newaxis] - points1, axis=2)
    dist_matrix_2 = np.linalg.norm(points2[:, np.newaxis] - points2, axis=2)
    dist_matrix_12 = np.linalg.norm(points1[:, np.newaxis] - points2, axis=2)

    # Define uniform weights for each point (uniform distribution assumption)
    mu1 = np.ones(n1) / n1
    mu2 = np.ones(n2) / n2

    # Compute the Gromov-Wasserstein distance
    gw_distance = ot.gromov_wasserstein(dist_matrix_1, dist_matrix_2, mu1, mu2, 'square_loss')

    # Extract a scalar value (e.g., the mean or min) from the distance array
    return np.mean(gw_distance)  # or np.min(gw_distance) depending on what you want

def find_loop_closures(bag1_path, bag2_path, threshold=0.5):
    """
    Find loop closures between two bags.
    """
    loop_closures = []
    for pcd1_file in os.listdir(bag1_path):
        if pcd1_file.endswith('.pcd'):
            pcd1_path = os.path.join(bag1_path, pcd1_file)
            cloud1 = o3d.io.read_point_cloud(pcd1_path)

            for pcd2_file in os.listdir(bag2_path):
                if pcd2_file.endswith('.pcd'):
                    pcd2_path = os.path.join(bag2_path, pcd2_file)
                    cloud2 = o3d.io.read_point_cloud(pcd2_path)

                    # Compute the Gromov-Wasserstein distance between the two scans
                    distance = compute_gromov_wasserstein(cloud1, cloud2)
                    if distance < threshold:
                        loop_closures.append((pcd1_file, pcd2_file, distance))
    return loop_closures


# Example usage
bag1_path = "/home/agrima/loop-closure-lpgw/data/preprocessed/instance1_pcd"
bag2_path = "/home/agrima/loop-closure-lpgw/data/preprocessed/instance2_pcd"
loop_closures = find_loop_closures(bag1_path, bag2_path, threshold=0.5)
print(f"Loop closures between Bag 1 and Bag 2: {loop_closures}")

