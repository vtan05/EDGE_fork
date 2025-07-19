import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from slice_finedance import slice_finedance


def get_finedance_split():
    # Generate all sample IDs from 001 to 211
    all_list = [str(i).zfill(3) for i in range(1, 212)]

    # Predefined test and ignore lists
    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193",
                 "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]

    # All non-test samples are in the train set
    train_list = [x for x in all_list if x not in test_list]

    return ignor_list, train_list, test_list


def create_dataset(opt):
    # Get split lists for FineDance
    print("Loading FineDance train/test split")
    ignor_list, train_list, test_list = get_finedance_split()

    # Slice motions and audio for training
    print("Slicing train data")
    slice_finedance(train_list, opt.dataset_folder, "train", stride=opt.stride, length=opt.length)

    # Slice motions and audio for testing
    print("Slicing test data")
    slice_finedance(test_list, opt.dataset_folder, "test", stride=opt.stride, length=opt.length)

    # baseline_extract("/host_data/van/EDGE/data/finedance/train/wavs_sliced", "/host_data/van/EDGE/data/finedance/train/baseline_feats")
    # baseline_extract("/host_data/van/EDGE/data/finedance/test/wavs_sliced", "/host_data/van/EDGE/data/finedance/test/baseline_feats")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5, help="Sliding window stride (seconds)")
    parser.add_argument("--length", type=float, default=5.0, help="Window length (seconds)")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="/host_data/van/EDGE/data/finedance",
        help="Folder containing raw motions and music",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
