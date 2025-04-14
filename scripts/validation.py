import os
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
original_dir = "/home/agrima/loop-closure-lpgw/mar19/testbag_Frames"  # Directory containing the original frames
subsampled_dir = "/media/agrima/Fusion/loop-closure-lpgw/mar19/subsampled_frames"  # Directory to save the subsampled frames
os.makedirs(subsampled_dir, exist_ok=True)

# Load all frame filenames
original_frames = sorted([f for f in os.listdir(original_dir) if f.endswith(".npy")])

# Subsampling factor (e.g., keep every 5th frame)
subsampling_factor = 5
subsampled_frames = original_frames[::subsampling_factor]

print(f"Reduced number of frames: {len(subsampled_frames)}")

# Split into train and test sets (e.g., 80% train, 20% test)
train_frames, test_frames = train_test_split(subsampled_frames, test_size=0.2, random_state=42)

# Create train and test directories
train_dir = os.path.join(subsampled_dir, "train")
test_dir = os.path.join(subsampled_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Save train and test frames
def save_frames(frames, output_dir):
    for frame in frames:
        src_path = os.path.join(original_dir, frame)
        dst_path = os.path.join(output_dir, frame)
        data = np.load(src_path)
        np.save(dst_path, data)

save_frames(train_frames, train_dir)
save_frames(test_frames, test_dir)

print(f"✅ Train frames saved in: {train_dir}")
print(f"✅ Test frames saved in: {test_dir}")