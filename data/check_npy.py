import numpy as np

# Replace with your actual path
npy_path = '/host_source/van/EDGE_fork/data/finedance/motion/001.npy'

# Load the .npy file
data = np.load(npy_path, allow_pickle=True)

# Print the type and contents
print("📦 Loaded type:", type(data))
print("✅ Contents:\n", data)


if isinstance(data, dict):
    print("🧭 Keys:", list(data.keys()))
    for key, value in data.items():
        print(f"\n🔑 {key}:")
        print("  🔹 Shape:", np.shape(value))
        print("  🔹 Type:", type(value))
        print("  🔹 First few values:\n", value if np.isscalar(value) else value[:5])
