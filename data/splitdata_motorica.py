import glob
import os
import pickle
import shutil
from pathlib import Path
import numpy as np


def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out


train_list = set(fileToList(r"/host_data/van/EDGE/data/motorica/train_files.txt"))
test_list = set(fileToList(r"/host_data/van/EDGE/data/motorica/test_files.txt"))


def split_data(dataset_path):
    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{split_name}/motions").mkdir(parents=True, exist_ok=True)
        Path(f"{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        for sequence in split_list:
            motion = f"{dataset_path}/motions/{sequence}.npz"
            wav = f"{dataset_path}/wavs/{sequence}.wav"
            assert os.path.isfile(motion)
            assert os.path.isfile(wav)
            motion_data = np.load(motion)
            trans = motion_data["trans"]
            pose = motion_data["poses"]
            scale = motion_data["scaling"]
            out_data = {"pos": trans, "q": pose, "scale": scale}
            pickle.dump(out_data, open(f"{split_name}/motions/{sequence}.pkl", "wb"))
            shutil.copyfile(wav, f"{split_name}/wavs/{sequence}.wav")
