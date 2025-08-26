import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract


# === Sort helper for slice filenames like song_slice01.wav ===
def stringintcmp_(a, b):
    get_key = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])
    prefix_a, prefix_b = "_".join(a.split("_")[:-1]), "_".join(b.split("_")[:-1])
    ka, kb = get_key(a), get_key(b)
    if prefix_a != prefix_b:
        return -1 if prefix_a < prefix_b else 1
    return -1 if ka < kb else (1 if ka > kb else 0)


stringintkey = cmp_to_key(stringintcmp_)


def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1  # Each slice = 2.5s
    temp_dir_list = []
    all_cond = []
    all_filenames = []

    if opt.use_cached_features:
        print("Using precomputed features")
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            wav_files = sorted(glob.glob(os.path.join(dir, "*.wav")), key=stringintkey)
            feat_files = sorted(glob.glob(os.path.join(dir, "*.npy")), key=stringintkey)
            assert len(wav_files) == len(feat_files), f"Mismatch in {dir}"

            if len(wav_files) < sample_size:
                print(f"Skipping {dir}: not enough slices")
                continue

            rand_idx = random.randint(0, len(wav_files) - sample_size)
            selected_feats = feat_files[rand_idx : rand_idx + sample_size]
            selected_wavs = wav_files[rand_idx : rand_idx + sample_size]

            cond_list = [np.load(f) for f in selected_feats]
            all_cond.append(torch.from_numpy(np.array(cond_list)))
            all_filenames.append(selected_wavs)

    else:
        print("Computing features for input music slices")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name

            # Slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)

            if len(file_list) < sample_size:
                print(f"Skipping {wav_file}: not enough slices")
                continue

            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []
            file_subset = []

            print(f"Computing features for slices of: {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                if not (rand_idx <= idx < rand_idx + sample_size):
                    continue
                reps, _ = feature_func(file)
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                cond_list.append(reps)
                file_subset.append(file)

            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_subset)

    # Load model
    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    fk_out = opt.motion_save_dir if opt.save_motions else None

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1,
            fk_out=fk_out, render=not opt.no_render
        )

    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
