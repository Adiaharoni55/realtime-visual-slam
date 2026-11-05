"""Repeated runs: how much do ATE and RPE vary by chance, and is a difference between configurations
bigger than that?

Usage: python scripts/noise_runs.py [name filter]   (run from the repo root, after building)

Runs every configuration below with --start 0..4: almost the same data, but the keyframes fall
differently. Prints mean ± std (and min..max) of ATE and RPE per configuration, plus the loops,
checked against the ground truth by scripts/check_loops.py (a loop with an error above 0.3 m or
10° counts as wrong). Raw numbers go to output/noise_runs.csv.
"""
import re
import subprocess
import sys

import numpy as np

STARTS = [0, 1, 2, 3, 4]
VOCAB = ["--vocab", "models/orb_vocab.bin"]
MASK = ["--mask", "models/deeplabv3_mobilenet.onnx"]
WALKING = ("data/rgbd_dataset_freiburg3_walking_xyz", "configs/tum_fr3.yaml")
SITTING = ("data/rgbd_dataset_freiburg3_sitting_xyz", "configs/tum_fr3.yaml")
STRUCTURE = ("data/rgbd_dataset_freiburg3_structure_texture_far", "configs/tum_fr3.yaml")

# (name, (sequence, camera), extra arguments)
CONFIGS = [
    ("walking odometry", WALKING, []),
    ("walking odometry + mask", WALKING, MASK),
    ("walking loops", WALKING, VOCAB),
    ("walking loops + mask", WALKING, VOCAB + MASK),
    ("sitting odometry", SITTING, []),
    ("sitting odometry + mask", SITTING, MASK),
    ("sitting loops", SITTING, VOCAB),
    ("sitting loops + mask", SITTING, VOCAB + MASK),
    ("structure odometry", STRUCTURE, []),
    ("structure odometry + mask", STRUCTURE, MASK),
    ("structure BA", STRUCTURE, ["--ba"]),
    ("structure loops", STRUCTURE, VOCAB),
    ("structure BA + loops", STRUCTURE, ["--ba"] + VOCAB),
]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def rmse(evo_output):
    return float(re.search(r"rmse\s+([\d.]+)", evo_output).group(1))


def loop_check(seq):
    """(loops, wrong loops) from check_loops.py's per-loop rows."""
    out = subprocess.run(["python", "scripts/check_loops.py", seq], capture_output=True, text=True)
    rows = re.findall(r"^\s*\d+\s+\d+\s+\d+\s+\S+m\s+\S+m\s+([\d.]+)m\s+([\d.]+)°", out.stdout, re.M)
    wrong = sum(float(t) > 0.3 or float(r) > 10 for t, r in rows)
    return len(rows), wrong


name_filter = sys.argv[1] if len(sys.argv) > 1 else ""
results = []   # (name, start, ate, rpe, loops, wrong)
for name, (seq, camera), extra in CONFIGS:
    if name_filter not in name:
        continue
    for start in STARTS:
        out = run(["./build/release/rtvm_run_vo", seq, camera, *extra, "--start", str(start)])
        failed = int(re.search(r"Failed frames: (\d+)", out).group(1))
        gt = f"{seq}/groundtruth.txt"
        ate = rmse(run(["evo_ape", "tum", gt, "output/trajectory.txt", "--align"]))
        rpe = rmse(run(["evo_rpe", "tum", gt, "output/trajectory.txt", "--align",
                        "--delta", "30", "--delta_unit", "f", "--all_pairs"]))
        loops, wrong = loop_check(seq) if "--vocab" in extra else (0, 0)
        results.append((name, start, ate, rpe, loops, wrong, failed))
        print(f"{name:<26} start {start}: ATE {100 * ate:6.2f} cm  RPE {100 * rpe:5.2f} cm  "
              f"loops {loops:3d} ({wrong} wrong)  failed {failed}", flush=True)

with open("output/noise_runs.csv", "w") as f:
    f.write("config,start,ate,rpe,loops,wrong_loops,failed\n")
    for r in results:
        f.write(",".join(str(x) for x in r) + "\n")


def spread(values):
    """'mean ± std (min..max)' in cm."""
    v = 100 * np.array(values)
    return f"{v.mean():5.2f} ± {v.std(ddof=1):4.2f} ({v.min():5.2f}..{v.max():5.2f})"


print(f"\n{'configuration':<26} {'ATE cm: mean ± std (min..max)':<32} "
      f"{'RPE cm: mean ± std (min..max)':<32} {'loops':>6} {'wrong':>6}")
for name, _, _ in CONFIGS:
    rows = [r for r in results if r[0] == name]
    if not rows:
        continue
    print(f"{name:<26} {spread([r[2] for r in rows]):<32} {spread([r[3] for r in rows]):<32} "
          f"{np.mean([r[4] for r in rows]):6.1f} {sum(r[5] for r in rows):6d}")
print("Saved output/noise_runs.csv")
