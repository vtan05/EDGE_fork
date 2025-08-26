import os
import librosa
import soundfile as sf
from pathlib import Path

test_list = [
    "063", "132", "143", "036", "098", "198", "130", "012", "211", "193",
    "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"
]

music_dir = "/host_data/van/EDGE/data/finedance/music_wav"     # 🔁 Change this
output_dir = "/host_data/van/EDGE/data/finedance/test/eval_34s"     # 🔁 Change this
slice_frames = 1024
fps = 30
slice_secs = slice_frames / fps  # ~34.13

os.makedirs(output_dir, exist_ok=True)

for song_id in test_list:
    wav_path = os.path.join(music_dir, f"{song_id}.wav")
    if not os.path.isfile(wav_path):
        print(f"❌ File not found: {wav_path}")
        continue

    y, sr = librosa.load(wav_path, sr=None)
    samples_per_slice = int(slice_secs * sr)
    num_slices = len(y) // samples_per_slice

    print(f"🔹 {song_id}.wav → {num_slices} slices of {slice_secs:.2f}s ({samples_per_slice} samples)")

    for i in range(num_slices):
        start = i * samples_per_slice
        end = start + samples_per_slice
        audio_slice = y[start:end]

        out_path = os.path.join(output_dir, f"{song_id}_slice{i}.wav")
        sf.write(out_path, audio_slice, sr)
