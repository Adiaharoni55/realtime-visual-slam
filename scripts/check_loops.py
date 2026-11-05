"""Accepted loop edges vs. ground truth: is each measured relative pose the true one?

Usage: python scripts/check_loops.py <TUM folder>   (after rtvm_run_vo --vocab)

For each row of output/loops.csv, compares the PnP-measured motion (keyframe `to` seen from
keyframe `from`) with the ground truth's relative pose T_from^-1 * T_to. A correct loop agrees
within a few centimeters and degrees; a false one (perceptual aliasing) doesn't.
"""
import sys

import numpy as np

seq = sys.argv[1]
stats = np.genfromtxt("output/stats.csv", delimiter=",", names=True)
loops = np.genfromtxt("output/loops.csv", delimiter=",", names=True, ndmin=1)
gt = np.loadtxt(f"{seq}/groundtruth.txt")

# Keyframe id k = the k-th row where a new keyframe was created.
kf_times = stats["timestamp"][stats["new_keyframe"] == 1]


def matrix(t, q):
    """4x4 pose from a translation and a quaternion (x, y, z, w)."""
    x, y, z, w = q / np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = t
    return T


def gt_pose(t, max_diff=0.02):
    """Ground-truth pose closest to time t, or None in a gap."""
    i = np.argmin(np.abs(gt[:, 0] - t))
    return matrix(gt[i, 1:4], gt[i, 4:8]) if abs(gt[i, 0] - t) <= max_diff else None


def angle_deg(R):
    """Rotation angle of a 3x3 rotation matrix, in degrees."""
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


if loops.size == 0:
    sys.exit("No loops in output/loops.csv")

print(f"{'from':>4} {'to':>4} {'inliers':>7} {'true dist':>9} {'meas dist':>9} "
      f"{'t error':>8} {'R error':>8}")
t_errors, r_errors = [], []
for row in loops:
    a, b = gt_pose(kf_times[int(row["from"])]), gt_pose(kf_times[int(row["to"])])
    if a is None or b is None:
        print(f"{int(row['from']):>4} {int(row['to']):>4}   (ground-truth gap, skipped)")
        continue
    true_rel = np.linalg.inv(a) @ b
    measured = matrix([row["tx"], row["ty"], row["tz"]], [row["qx"], row["qy"], row["qz"], row["qw"]])
    error = np.linalg.inv(true_rel) @ measured   # identity if the loop edge is exactly right
    t_errors.append(np.linalg.norm(error[:3, 3]))
    r_errors.append(angle_deg(error[:3, :3]))
    print(f"{int(row['from']):>4} {int(row['to']):>4} {int(row['inliers']):>7} "
          f"{np.linalg.norm(true_rel[:3, 3]):>8.3f}m {np.linalg.norm(measured[:3, 3]):>8.3f}m "
          f"{t_errors[-1]:>7.3f}m {r_errors[-1]:>7.2f}°")

print(f"\n{len(t_errors)} loops: translation error median {np.median(t_errors):.3f} m, "
      f"max {np.max(t_errors):.3f} m; rotation error median {np.median(r_errors):.2f}°, "
      f"max {np.max(r_errors):.2f}°")
