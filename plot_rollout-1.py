#!/usr/bin/env python3
"""
plot_rollout.py — Group HW2 Graph G2: joint trajectories of one policy rollout episode.

Usage (on the machine where you ran the policy, i.e. where the eval dataset lives):
    python plot_rollout.py <hf-id>/eval_pickplace_v1                 # episode 0
    python plot_rollout.py <hf-id>/eval_pickplace_v1 --episode 3
    python plot_rollout.py /path/to/dataset/folder --episode 3

Optional (Part 2): overlay one human teleop demonstration for comparison
    python plot_rollout.py <hf-id>/eval_pickplace_v1 --episode 3 \
        --compare <hf-id>/so101_pickplace_v1 --compare-episode 0

Reads the LeRobot dataset's parquet files, takes `observation.state`
(the 6 measured joint values) for the chosen episode, and draws one small
panel per joint vs. time. Six panels, one figure. Nothing else to configure.
"""
import argparse
import glob
import json
import os
import sys

JOINT_FALLBACK = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]


def resolve_dataset(spec):
    """Return a local directory for either a path or a HF repo id."""
    if os.path.isdir(spec):
        return spec
    roots = [
        os.environ.get("HF_LEROBOT_HOME"),
        os.path.join(os.environ.get("HF_HOME", ""), "lerobot") if os.environ.get("HF_HOME") else None,
        os.path.expanduser("~/.cache/huggingface/lerobot"),
    ]
    for r in roots:
        if r and os.path.isdir(os.path.join(r, spec)):
            return os.path.join(r, spec)
    # not local — try the Hub
    try:
        from huggingface_hub import snapshot_download
        print(f"[GHW2] {spec} not found locally, downloading from the Hub ...")
        return snapshot_download(spec, repo_type="dataset")
    except Exception as e:
        print(f"[GHW2] stopped: could not find dataset '{spec}' locally or on the Hub ({e}).")
        sys.exit(1)


def load_episode(root, episode):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        print(f"[GHW2] stopped: no parquet files under {root}/data — is this a LeRobot dataset?")
        sys.exit(1)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "episode_index" not in df or "observation.state" not in df:
        print("[GHW2] stopped: parquet has no 'episode_index' / 'observation.state' columns.")
        sys.exit(1)
    ep = df[df["episode_index"] == episode].sort_values("frame_index" if "frame_index" in df else "timestamp")
    if ep.empty:
        avail = sorted(df["episode_index"].unique().tolist())
        print(f"[GHW2] stopped: episode {episode} not in dataset. Available: {avail[:20]}{' ...' if len(avail) > 20 else ''}")
        sys.exit(1)
    import numpy as np
    state = np.stack(ep["observation.state"].to_numpy())
    t = ep["timestamp"].to_numpy() if "timestamp" in ep else np.arange(len(ep)) / 30.0
    t = t - t[0]
    return t, state


def joint_names(root, n):
    try:
        with open(os.path.join(root, "meta", "info.json")) as f:
            names = json.load(f)["features"]["observation.state"]["names"]
        if len(names) == n:
            return [str(x) for x in names]
    except Exception:
        pass
    return JOINT_FALLBACK[:n] if n <= 6 else [f"joint {i}" for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="HF repo id (e.g. myid/eval_pickplace_v1) or local folder")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--compare", default=None, help="teleop dataset to overlay (Part 2)")
    ap.add_argument("--compare-episode", type=int, default=0)
    ap.add_argument("-o", "--out", default="rollout_joints.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    root = resolve_dataset(args.dataset)
    t, s = load_episode(root, args.episode)
    names = joint_names(root, s.shape[1])

    cmp = None
    if args.compare:
        croot = resolve_dataset(args.compare)
        cmp = load_episode(croot, args.compare_episode)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = s.shape[1]
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), dpi=150, sharex=True)
    axes = axes.ravel()
    for j in range(n):
        ax = axes[j]
        if cmp is not None:
            ax.plot(cmp[0], cmp[1][:, j], color="#999999", linewidth=1.2, label="teleop demo")
        ax.plot(t, s[:, j], color="#0F4C5C", linewidth=1.6, label="policy rollout")
        ax.set_title(names[j], fontsize=10, color="#E36414", loc="left")
        ax.grid(alpha=0.3)
        if j >= n - cols:
            ax.set_xlabel("time [s]")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    if cmp is not None:
        axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(args.title or f"Policy rollout — episode {args.episode}  ({len(t)} frames, {t[-1]:.1f} s)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[GHW2] episode {args.episode}: {len(t)} frames, {t[-1]:.1f} s, {n} joints")
    print(f"[GHW2] saved {args.out}")


if __name__ == "__main__":
    main()
