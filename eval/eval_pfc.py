import argparse
import glob
import os
import pickle

import numpy as np
from tqdm import tqdm


def calc_physical_score(dir):
    scores = []
    names = []
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30

    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))
        print(np.min(info["full_pose"]), np.max(info["full_pose"]))
        joint3d = info["full_pose"]
        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )  # (S-2, 4) horizontal velocity
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    print(f"{dir} has a mean PFC of {out}")

# === Jitter from PKL motion ===
def measure_jitter(joint_pos, fps):
    vel = (joint_pos[1:] - joint_pos[:-1]) * fps
    acc = (vel[1:] - vel[:-1]) * fps
    jitter = np.mean(np.linalg.norm(acc, axis=-1))
    return jitter


def measure_jitter_from_pkl(dir: str, fps: int = 30):
    print("Computing jitter metric from .pkl files:")
    file_list = glob.glob(os.path.join(dir, "*.pkl"))
    if len(file_list) > 1000:
        file_list = random.sample(file_list, 1000)

    total_jitter = np.zeros([len(file_list)])

    for i, fname in enumerate(tqdm(file_list, desc="Computing Jitter")):
        info = pickle.load(open(fname, "rb"))
        joint_pos = info["full_pose"] #* 0.01  # Convert to meters
        jitter = measure_jitter(joint_pos, fps)
        total_jitter[i] = jitter

    jitter_mean = total_jitter.mean()
    print(f"Total mean jitter of {len(file_list)} motions: {jitter_mean:.6f}")


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="/host_data/van/EDGE/results/finedance/",
        help="Where to load saved motions",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS of the motion data (used in jitter calc)",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    calc_physical_score(opt.motion_path)
    measure_jitter_from_pkl(opt.motion_path, opt.fps)
