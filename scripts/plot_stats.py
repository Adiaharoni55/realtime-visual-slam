"""Plot per-frame tracking stats from rtvm_run_vo (output/stats.csv)."""
import sys

import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "output/stats.csv"
d = np.genfromtxt(path, delimiter=",", names=True)
t = d["timestamp"] - d["timestamp"][0]          # seconds since the start

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, d["matches"], label="matches", lw=0.8)
ax.plot(t, d["inliers"], label="inliers", lw=0.8)
kf = d["new_keyframe"] == 1
ax.scatter(t[kf], d["inliers"][kf], s=8, c="k", label="new keyframe", zorder=3)
fail = d["success"] == 0
ax.scatter(t[fail], np.zeros(fail.sum()), s=30, c="r", marker="x", label="failure", zorder=4)
ax.set_xlabel("time (s)")
ax.set_ylabel("count")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/stats.png", dpi=150)
print("Saved output/stats.png")

# The 5 weakest frames (fewest inliers, first frame excluded), to inspect the images.
order = np.argsort(d["inliers"][1:])[:5] + 1
print("\nFewest inliers:")
for i in order:
    print(f"  t={t[i]:6.2f}s  inliers={int(d['inliers'][i]):4d}  matches={int(d['matches'][i]):4d}"
          f"  image: rgb/{d['timestamp'][i]:.6f}.png")
plt.show()