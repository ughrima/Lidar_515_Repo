from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.manifold import MDS
import numpy as np
import os

# Compute pairwise Euclidean distances
def compute_distance_matrix(pc):
    distances = pdist(pc, metric="euclidean")  # Compute pairwise distances
    distance_matrix = squareform(distances).astype(np.float32)  # Convert to square matrix
    return distance_matrix

# Convert distance matrix to Gram matrix
def compute_gram_matrix(D):
    """ Converts a distance matrix to a Gram matrix using double centering. """
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    G = -0.5 * H @ (D ** 2) @ H  # Gram matrix
    return G

# Check if distance matrix is Euclidean
def is_valid_euclidean(D, tol=1e-6):
    """ Checks if the distance matrix is Euclidean by examining the Gram matrix eigenvalues. """
    G = compute_gram_matrix(D)
    eigvals, eigvecs = eigh(G)
    min_eig = np.min(eigvals)

    if min_eig < -tol:  # Significant negative eigenvalues → Fix needed
        print(f"⚠️ Invalid Euclidean matrix! Min eigenvalue: {min_eig:.6f}")
        return False
    elif min_eig < 0:  # Small numerical errors → Clip them
        print(f"🔍 Small numerical error detected (Min eigenvalue: {min_eig:.6f}). Clipping...")
        eigvals[eigvals < 0] = 0  # Set negative eigenvalues to zero
        G_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T  # Reconstruct Gram matrix
        D_fixed = np.sqrt(np.maximum(0, np.diag(G_fixed)[:, None] + np.diag(G_fixed) - 2 * G_fixed))
        return True  # Considered valid after clipping
    else:
        print(f"✅ Valid Euclidean matrix. Min eigenvalue: {min_eig:.6f}")
        return True

# Fix non-Euclidean distance matrix using MDS
def project_to_euclidean(D):
    """ Projects the given distance matrix to the closest valid Euclidean space using MDS. """
    print("🔄 Projecting distance matrix to Euclidean space using MDS...")
    n_components = min(D.shape[0], 256)  # Preserve as much structure as possible
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=42, n_jobs=1)
    new_points = mds.fit_transform(D)

    # Compute new Euclidean distances
    D_fixed = np.sqrt(((new_points[:, np.newaxis] - new_points) ** 2).sum(axis=2))
    
    print("✅ Distance matrix successfully projected to valid Euclidean space.")
    return D_fixed

# Process input directories and compute valid distance matrices
def process_frames(input_dir, output_dir):
    files = sorted(os.listdir(input_dir))
    for i, frame in enumerate(files):
        if frame.endswith(".npy"):
            pc = np.load(os.path.join(input_dir, frame)).astype(np.float32)

            if len(pc) < 2:
                print(f"⚠️ Skipping {frame}: Not enough points ({len(pc)})")
                continue
            
            # Compute distance matrix
            D = compute_distance_matrix(pc)

            # Validate Euclidean property
            if not is_valid_euclidean(D):
                print(f"⚠️ Fixing distance matrix for {frame}...")
                D = project_to_euclidean(D)  # Fix it

            np.save(os.path.join(output_dir, f"distance_matrix_{i}.npy"), D)

# Paths
train_dir = "/home/agrima/loop-closure-lpgw/mar19/downsample_frames/train"
test_dir = "/home/agrima/loop-closure-lpgw/mar19/downsample_frames/test"
distances_dir = "/home/agrima/loop-closure-lpgw/mar19/distance_matrices"
train_output_dir = os.path.join(distances_dir, "train")
test_output_dir = os.path.join(distances_dir, "test")
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Run processing
process_frames(train_dir, train_output_dir)
process_frames(test_dir, test_output_dir)

print(f"✅ All distance matrices are saved in: {distances_dir}")


# icp,ffph,bag of words, deep learning 
