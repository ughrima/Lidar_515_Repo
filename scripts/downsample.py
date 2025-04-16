from scipy.linalg import eigh
from sklearn.decomposition import PCA
import numpy as np
import os

# Convert distance matrix to Gram matrix
def compute_gram_matrix(D):
    n = D.shape[0]
    H = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    G = -0.5 * H @ (D.astype(np.float64) ** 2) @ H
    return G

# Check if distance matrix is Euclidean
def is_valid_euclidean(D, tol=1e-6):
    G = compute_gram_matrix(D)
    eigvals, _ = eigh(G)
    min_eig = np.min(eigvals)

    if min_eig < -tol:
        print(f"⚠️ Invalid Euclidean matrix! Min eigenvalue: {min_eig:.6f}")
        return False
    elif min_eig < 0:
        print(f"🔍 Small numerical error detected (Min eigenvalue: {min_eig:.6f}). Clipping...")
        eigvals[eigvals < 0] = 0
        G_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        D_fixed = np.sqrt(np.maximum(0, np.diag(G_fixed)[:, None] + np.diag(G_fixed) - 2 * G_fixed))
        return True
    else:
        print(f"✅ Valid Euclidean matrix. Min eigenvalue: {min_eig:.6f}")
        return True

# Fix non-Euclidean distance matrix using PCA
def project_to_euclidean_pca(D):
    print("🔄 Projecting distance matrix to Euclidean space using PCA...")
    n_components = min(D.shape[0], 128)
    pca = PCA(n_components=n_components, random_state=42)
    new_points = pca.fit_transform(D)
    D_fixed = np.sqrt(((new_points[:, np.newaxis] - new_points) ** 2).sum(axis=2))
    print("✅ Distance matrix successfully projected to valid Euclidean space.")
    return D_fixed

# Process frames and validate/fix distance matrices
def process_frames(input_dir, output_dir):
    files = sorted(os.listdir(input_dir))
    for frame in files:
        if frame.endswith(".npy"):
            D = np.load(os.path.join(input_dir, frame)).astype(np.float32)
            
            if not is_valid_euclidean(D):
                print(f"⚠️ Fixing distance matrix for {frame}...")
                D = project_to_euclidean_pca(D)
            
            np.save(os.path.join(output_dir, frame), D)

# Paths
train_input_dir = "/media/agrima/Fusion/loop-closure-lpgw/distance/train"
test_input_dir = "/media/agrima/Fusion/loop-closure-lpgw/distance/test"
fixed_matrices_dir = "/media/agrima/Fusion/loop-closure-lpgw/fixed_distance_matrices"
train_output_dir = os.path.join(fixed_matrices_dir, "train")
test_output_dir = os.path.join(fixed_matrices_dir, "test")
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Validate and fix distance matrices for train and test frames
process_frames(train_input_dir, train_output_dir)
process_frames(test_input_dir, test_output_dir)

print("✅ Fixed distance matrices saved successfully.")