import numpy as np
import os
import shutil

frames_dir = "/home/agrima/loop-closure-lpgw/mar19/testbag_Frames"
output_dir = "/home/agrima/loop-closure-lpgw/mar19/output_splits"
os.makedirs(output_dir, exist_ok=True)

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".npy")])
split_ratio = 0.8
num_train = int(len(all_frames) * split_ratio)

train_frames = all_frames[:num_train]
test_frames = all_frames[num_train:]

np.save(os.path.join(output_dir, "train_frames.npy"), train_frames)
np.save(os.path.join(output_dir, "test_frames.npy"), test_frames)

for frame in train_frames:
    shutil.copy(os.path.join(frames_dir, frame), os.path.join(train_dir, frame))

for frame in test_frames:
    shutil.copy(os.path.join(frames_dir, frame), os.path.join(test_dir, frame))

print(f"✅ Split {len(all_frames)} frames: {len(train_frames)} train, {len(test_frames)} test.")
