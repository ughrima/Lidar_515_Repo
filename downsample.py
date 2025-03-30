import open3d as o3d
import numpy as np
import os

# Function to downsample a point cloud
def downsample_point_cloud(pc, target_points=256):
    """Downsample point cloud to exactly target_points using Open3D voxel sampling.
       If too few points exist, pad with repeated points."""
    
    # Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    
    # Voxel downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.3)
    
    # Convert back to NumPy array
    downsampled_pc = np.asarray(pcd.points)
    
    # Ensure exactly 'target_points' count
    if downsampled_pc.shape[0] > target_points:
        indices = np.random.choice(downsampled_pc.shape[0], target_points, replace=False)
        downsampled_pc = downsampled_pc[indices]
    elif downsampled_pc.shape[0] < target_points:
        extra_points = downsampled_pc[np.random.choice(downsampled_pc.shape[0], target_points - downsampled_pc.shape[0], replace=True)]
        downsampled_pc = np.vstack((downsampled_pc, extra_points))

    return downsampled_pc

# Directories for training and testing frames
train_dir = "/home/agrima/loop-closure-lpgw/mar19/output_splits/train"
test_dir = "/home/agrima/loop-closure-lpgw/mar19/output_splits/test"

# Output directories for downsampled frames
downsampled_dir = "/home/agrima/loop-closure-lpgw/mar19/scripts/downsample_frames"
train_output_dir = os.path.join(downsampled_dir, "train")
test_output_dir = os.path.join(downsampled_dir, "test")
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Downsample and save point clouds
def process_frames(input_dir, output_dir, target_points=256):
    files = sorted(os.listdir(input_dir))
    for i, frame in enumerate(files):
        if frame.endswith(".npy"):
            pc = np.load(os.path.join(input_dir, frame))

            if pc.shape[0] < 2:
                print(f"⚠️ Skipping {frame}: Not enough points ({pc.shape[0]})")
                continue

            downsampled_pc = downsample_point_cloud(pc, target_points)
            np.save(os.path.join(output_dir, f"frame_{i}.npy"), downsampled_pc)

process_frames(train_dir, train_output_dir)
process_frames(test_dir, test_output_dir)

print(f"✅ Fixed downsampling. Frames saved in: {downsampled_dir}")
