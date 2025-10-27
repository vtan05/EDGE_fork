import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def _num_slices(total_len: int, window: int, stride: int) -> int:
    """Integer-safe slice count: number of windows of size `window`
    that fit into a sequence of length `total_len` with step `stride`."""
    if total_len < window:
        return 0
    return 1 + (total_len - window) // stride


def slice_audio(audio_file, stride, length, out_dir, max_slices=None):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    window = int(round(length * sr))
    stride_step = int(round(stride * sr))

    n_total = _num_slices(len(audio), window, stride_step)
    n = n_total if max_slices is None else min(n_total, max_slices)

    for i in range(n):
        start = i * stride_step
        audio_slice = audio[start: start + window]
        sf.write(f"{out_dir}/{file_name}_slice{i}.wav", audio_slice, sr)

    return n


def slice_motion(motion_file, stride, length, num_slices, out_dir, fps=30):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    # normalize root position
    pos = pos / scale

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    window = int(round(length * fps))
    stride_step = int(round(stride * fps))

    n_total = _num_slices(len(pos), window, stride_step)
    n = min(n_total, int(num_slices))

    for i in range(n):
        start = i * stride_step
        pos_slice = pos[start: start + window]
        q_slice = q[start: start + window]
        out = {"pos": pos_slice, "q": q_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{i}.pkl", "wb"))

    return n


def slice_motorica(motion_dir, wav_dir, stride=0.5, length=5, fps=30):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)

    for wav, motion in tqdm(zip(wavs, motions), total=len(wavs)):
        # names must match
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))

        # --- pre-compute common slice count without writing anything yet
        # audio slice count
        audio_sig, sr = lr.load(wav, sr=None)
        window_a = int(round(length * sr))
        stride_a = int(round(stride * sr))
        n_audio = _num_slices(len(audio_sig), window_a, stride_a)

        # motion slice count
        mot = pickle.load(open(motion, "rb"))
        pos = mot["pos"] / mot["scale"][0]
        window_m = int(round(length * fps))
        stride_m = int(round(stride * fps))
        n_motion = _num_slices(len(pos), window_m, stride_m)

        n_common = min(n_audio, n_motion)

        # slice both to exactly the same count
        audio_slices = slice_audio(wav, stride, length, wav_out, max_slices=n_common)
        motion_slices = slice_motion(motion, stride, length, n_common, motion_out, fps=fps)

        # sanity check (should match now)
        assert audio_slices == motion_slices == n_common, str(
            (wav, motion, audio_slices, motion_slices, n_common)
        )


def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs, total=len(wavs)):
        slice_audio(wav, stride, length, wav_out)
