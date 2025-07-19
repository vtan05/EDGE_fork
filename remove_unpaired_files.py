import os

# Set your folder paths here
wav_folder = "/host_data/van/EDGE/data/finedance/test/wavs_sliced"
npy_folder = "/host_data/van/EDGE/data/finedance/test/baseline_feats"

# Get base filenames (without extension)
wav_basenames = {os.path.splitext(f)[0] for f in os.listdir(wav_folder) if f.endswith(".wav")}
npy_basenames = {os.path.splitext(f)[0] for f in os.listdir(npy_folder) if f.endswith(".npy")}

# Identify unpaired files
unpaired_wav = wav_basenames - npy_basenames
unpaired_npy = npy_basenames - wav_basenames

# Delete unpaired .wav files
for name in unpaired_wav:
    file_path = os.path.join(wav_folder, name + ".wav")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted unpaired wav: {file_path}")

# Delete unpaired .npy files
for name in unpaired_npy:
    file_path = os.path.join(npy_folder, name + ".npy")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted unpaired NPY: {file_path}")
