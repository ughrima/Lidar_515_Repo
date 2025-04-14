from scipy.spatial.distance import pdist, squareform
import numpy as np
import os

# Compute pairwise Euclidean distances
def compute_distance_matrix(pc):
    distances = pdist(pc, metric="euclidean")
    distance_matrix = squareform(distances).astype(np.float32)
    return distance_matrix

# Process frames and save distance matrices
def process_frames(input_dir, output_dir):
    files = sorted(os.listdir(input_dir))
    for frame in files:
        if frame.endswith(".npy"):
            pc = np.load(os.path.join(input_dir, frame)).astype(np.float32)
            
            if len(pc) < 2:
                print(f"⚠️ Skipping {frame}: Not enough points ({len(pc)})")
                continue
            
            D = compute_distance_matrix(pc)
            np.save(os.path.join(output_dir, frame), D)

# Paths
train_input_dir = "/media/agrima/Fusion/loop-closure-lpgw/downsample/train"
test_input_dir = "/media/agrima/Fusion/loop-closure-lpgw/downsample/test"
distance_matrices_dir = "/media/agrima/Fusion/loop-closure-lpgw/distance"
train_output_dir = os.path.join(distance_matrices_dir, "train")
test_output_dir = os.path.join(distance_matrices_dir, "test")
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Compute distance matrices for train and test frames
process_frames(train_input_dir, train_output_dir)
process_frames(test_input_dir, test_output_dir)

print("✅ Distance matrices saved successfully.")