#!/usr/bin/env python3
"""
plot_loss.py — Group HW2 Graph G1: ACT training loss curve from a Mill job log.

Usage:
    python plot_loss.py mill-123456.out            # writes loss_curve.png
    python plot_loss.py mill-123456.out -o g1.png  # custom output name

Reads the lerobot-train log (the .out file Slurm wrote), pulls every
"step:N ... loss:X" line, and plots loss vs. step. Nothing to configure.
"""
import argparse
import re
import sys

LINE_RE = re.compile(r"step:(\d+)\s.*?\bloss:([0-9.]+)")


def parse_log(path):
    steps, losses = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Slurm output file, e.g. mill-123456.out")
    ap.add_argument("-o", "--out", default="loss_curve.png")
    ap.add_argument("--title", default=None, help="Optional title, e.g. 'Team 3 — ACT v1'")
    args = ap.parse_args()

    steps, losses = parse_log(args.logfile)
    if not steps:
        print(f"[GHW2] stopped: no 'step:N ... loss:X' lines found in {args.logfile}.")
        print("      Is this the .out file from lerobot-train? Did the job get past step 200?")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(steps, losses, color="#0F4C5C", linewidth=1.8)
    ax.scatter(steps[-1], losses[-1], color="#E36414", zorder=3)
    ax.annotate(f"final loss = {losses[-1]:.3f}", (steps[-1], losses[-1]),
                xytext=(-10, 12), textcoords="offset points", ha="right",
                color="#E36414", fontsize=9)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(args.title or f"ACT training loss  ({len(steps)} log points, {steps[-1]:,} steps)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[GHW2] {len(steps)} points, steps {steps[0]}–{steps[-1]}, "
          f"loss {losses[0]:.3f} → {losses[-1]:.3f}")
    print(f"[GHW2] saved {args.out}")


if __name__ == "__main__":
    main()
