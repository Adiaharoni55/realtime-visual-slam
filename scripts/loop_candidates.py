"""Loop candidates vs. ground truth: does a high BoW score mean a real revisit?

Usage: python scripts/loop_candidates.py <TUM folder>   (after rtvm_run_vo --vocab)
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

seq = sys.argv[1]
stats = np.genfromtxt("output/stats.csv", delimiter=",", names=True)
gt = np.loadtxt(f"{seq}/groundtruth.txt")

# Keyframe id k = the k-th row where a new keyframe was created.
kf_times = stats["timestamp"][stats["new_keyframe"] == 1]


def gt_pose(t, max_diff=0.02):
    """Ground-truth position and quaternion (x, y, z, w) closest to time t, or None in a gap."""
    i = np.argmin(np.abs(gt[:, 0] - t))
    return (gt[i, 1:4], gt[i, 4:8]) if abs(gt[i, 0] - t) <= max_diff else None


scores, dists, angles, new_pos, old_pos = [], [], [], [], []
for row in stats[stats["loop_candidate"] >= 0]:
    now = gt_pose(row["timestamp"])
    old = gt_pose(kf_times[int(row["loop_candidate"])])
    if now is None or old is None:
        continue  # one of the two moments falls in a ground-truth gap
    scores.append(row["loop_score"])
    dists.append(np.linalg.norm(now[0] - old[0]))
    # Angle between the two camera orientations, from the quaternions.
    dot = min(1.0, abs(np.dot(now[1], old[1])))
    angles.append(np.degrees(2 * np.arccos(dot)))
    new_pos.append(now[0])
    old_pos.append(old[0])


scores, dists, angles = map(np.array, (scores, dists, angles))
real = (dists < 0.3) & (angles < 30)  # "same place": within 30 cm and 30 degrees

print(f"{len(scores)} candidates with ground truth, {real.sum()} real revisits")
if real.any():
    print(f"  real revisits: score {scores[real].min():.4f} .. {scores[real].max():.4f}")
if (~real).any():
    print(f"  not revisits:  score {scores[~real].min():.4f} .. {scores[~real].max():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(dists, scores, c=angles, cmap="viridis", s=15)
fig.colorbar(sc, label="angle between views (deg)")
ax.axvline(0.3, color="gray", ls="--", lw=0.8)
ax.set_xlabel("true distance between the two cameras (m)")
ax.set_ylabel("BoW score")
ax.set_title("Loop candidates vs. ground truth")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/loop_candidates.png", dpi=150)
print("Saved output/loop_candidates.png")

now_pos, old_pos = np.array(new_pos), np.array(old_pos)
accepted = scores >= 0.07

fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.plot(gt[:, 1], gt[:, 2], color="gray", lw=0.8, label="ground truth")
for a, b in zip(now_pos[accepted], old_pos[accepted]):
    ax2.plot([a[0], b[0]], [a[1], b[1]], color="red", lw=1.5)
ax2.plot([], [], color="red", label=f"loop candidates with score >= 0.07 ({accepted.sum()})")
ax2.scatter(*gt[0, 1:3], c="green", s=40, zorder=3, label="start")
ax2.scatter(*gt[-1, 1:3], c="black", s=40, zorder=3, label="end")
ax2.set_aspect("equal")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_title("Accepted loop candidates on the true path")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("output/loop_pairs.png", dpi=150)
print("Saved output/loop_pairs.png")

plt.show()