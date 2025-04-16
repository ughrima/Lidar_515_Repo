import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 Set folder path
folder_path = "/home/agrima/loop-closure-lpgw/mar19/distance_matrices/test"

# 🔹 Create a log file to store issues
log_file = os.path.join(folder_path, "matrix_quality_report.txt")

# 🔹 Function to load and ensure correct dtype
def load_matrix(file_path):
    D = np.load(file_path)
    if D.dtype == np.float16:
        D = D.astype(np.float32)  # ✅ Convert float16 to float32
    return D

# 🔹 Function to check if the matrix is square
def check_square(D, filename):
    if D.shape[0] != D.shape[1]:
        return f"❌ {filename}: Not square {D.shape}"
    return None

# 🔹 Function to check symmetry
def check_symmetry(D, filename):
    if not np.allclose(D, D.T, atol=1e-6):
        return f"⚠️ {filename}: Not symmetric"
    return None

# 🔹 Function to check non-negative values
def check_non_negative(D, filename):
    if np.any(D < 0):
        return f"⚠️ {filename}: Contains negative values"
    return None

# 🔹 Function to check NaN or Inf
def check_nan_inf(D, filename):
    if np.isnan(D).any() or np.isinf(D).any():
        return f"❌ {filename}: Contains NaN/Inf"
    return None

# 🔹 Function to check eigenvalues (PSD property)
def check_eigenvalues(D, filename):
    try:
        eigvals = np.linalg.eigvals(D)
        if np.min(eigvals) < -1e-5:  # Allow small numerical errors
            return f"⚠️ {filename}: Has negative eigenvalues (Min: {min(eigvals)})"
    except Exception as e:
        return f"❌ {filename}: Eigenvalue computation failed - {str(e)}"
    return None

# 🔹 Function to visualize a sample matrix
def plot_distance_matrix(D, filename):
    plt.figure(figsize=(6,6))
    sns.heatmap(D, cmap="viridis", square=True)
    plt.title(f"Matrix: {filename}")
    plt.show()

# 🔹 Process all matrices in the folder
issues = []
matrix_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".npy")]

for i, filename in enumerate(matrix_files):
    file_path = os.path.join(folder_path, filename)
    
    try:
        D = load_matrix(file_path)  # ✅ Now correctly loads float16 and converts to float32
        
        # 🔍 Run all checks
        issues.extend(filter(None, [
            check_square(D, filename),
            check_symmetry(D, filename),
            check_non_negative(D, filename),
            check_nan_inf(D, filename),
            check_eigenvalues(D, filename)
        ]))
        
        # 🔥 Visualize 3 random matrices
        if i % (len(matrix_files) // 3 + 1) == 0:
            plot_distance_matrix(D, filename)

    except Exception as e:
        issues.append(f"❌ {filename}: Error loading - {str(e)}")

# 🔹 Save report
with open(log_file, "w") as f:
    for issue in issues:
        f.write(issue + "\n")

print(f"✅ Matrix quality check complete. Report saved to {log_file}")
