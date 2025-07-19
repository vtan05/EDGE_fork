import os
import numpy as np
import soundfile as sf
import librosa as lr
from tqdm import tqdm
import pickle


def slice_audio(audio_file, stride, length, out_dir):
    audio, sr = lr.load(audio_file, sr=None)
    fname = os.path.splitext(os.path.basename(audio_file))[0]
    window = int(length * sr)
    stride_step = int(stride * sr)
    idx = 0
    start = 0
    while start + window <= len(audio):
        audio_slice = audio[start: start + window]
        sf.write(f"{out_dir}/{fname}_slice{idx}.wav", audio_slice, sr)
        start += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    data = np.load(motion_file)  # shape: (frames, features)
    fname = os.path.splitext(os.path.basename(motion_file))[0]
    window = int(length * 30)  # 30 FPS
    stride_step = int(stride * 30)
    idx = 0
    start = 0

    if data.shape[1] == 319:
        extract_pos = lambda x: x[:, 4:7]
        extract_q = lambda x: x[:, 7:]
    elif data.shape[1] == 315:
        extract_pos = lambda x: x[:, :3]
        extract_q = lambda x: x[:, 3:]
    else:
        print(f"⚠️ Skipping {fname}: unexpected shape {data.shape}")
        return 0

    while start + window <= len(data) and idx < num_slices:
        motion_slice = data[start: start + window]
        pos = extract_pos(motion_slice)  # (window, 3)
        q = extract_q(motion_slice)      # (window, 312 expected)

        if q.shape[1] != 312:
            print(f"⚠️ Skipping slice {idx} from {fname}: q.shape={q.shape}")
            start += stride_step
            continue

        # Save .npy slice
        out_npy = os.path.join(out_dir, f"{fname}_slice{idx}.npy")
        np.save(out_npy, motion_slice)

        # Save compatible .pkl format
        pkl_out = {
            "pos": pos,
            "q": q,
            "scale": [1.0]  # already normalized
        }
        out_pkl = os.path.join(out_dir, f"{fname}_slice{idx}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(pkl_out, f)

        start += stride_step
        idx += 1

    return idx


def slice_finedance(id_list, dataset_root, split_name, stride=0.5, length=5.0):
    motion_root = os.path.join(dataset_root, "motion")
    audio_root = os.path.join(dataset_root, "music_wav")

    motion_out = os.path.join(dataset_root, split_name, "motion_sliced")
    audio_out = os.path.join(dataset_root, split_name, "wavs_sliced")
    os.makedirs(motion_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)

    for id_ in tqdm(id_list, desc=f"[{split_name.upper()}] Slicing IDs"):
        motion_path = os.path.join(motion_root, f"{id_}.npy")
        audio_path = os.path.join(audio_root, f"{id_}.wav")

        if not os.path.exists(motion_path) or not os.path.exists(audio_path):
            print(f"⚠️ Skipping {id_}: motion or audio file not found")
            continue

        audio_slices = slice_audio(audio_path, stride, length, audio_out)
        motion_slices = slice_motion(motion_path, stride, length, audio_slices, motion_out)

        if audio_slices != motion_slices:
            print(f"❗ Mismatch for {id_}: audio={audio_slices}, motion={motion_slices}")
