#!/usr/bin/env python3
"""
fix_policy_stats.py — add normalization stats to a Mill-trained (LeRobot 0.5) ACT model
so the robot laptop (LeRobot 0.3.x) can run it.

Symptom this fixes (on the laptop, at the first inference step):
    Missing key(s) when loading model: {'normalize_inputs.buffer_observation_state.mean', ...}
    AssertionError: `mean` is infinity.

Run on the ROBOT LAPTOP, in your lerobot env, after `hf auth login`:
    python fix_policy_stats.py --model <hf-id>/act_so101_pickplace_v1 --dataset <hf-id>/so101_pickplace_v1

It downloads the model, reads mean/std from the dataset's meta/stats.json, writes the ten
missing tensors into model.safetensors, and re-uploads the model. Then run §6 as before.
"""
import argparse, json, os, sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download, upload_folder
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="e.g. TeamIORobotics/act_so101_pickplace_v1")
ap.add_argument("--dataset", required=True, help="e.g. TeamIORobotics/so101_pickplace_v1 (the v2.1 dataset you recorded)")
ap.add_argument("--no-upload", action="store_true", help="only patch the local copy")
args = ap.parse_args()

model_dir = snapshot_download(args.model)
stats_path = hf_hub_download(args.dataset, "meta/stats.json", repo_type="dataset")
stats = json.load(open(stats_path))
sd_path = os.path.join(model_dir, "model.safetensors")
sd = load_file(sd_path)

def t(key, field):
    if key not in stats:
        sys.exit(f"[fix] stopped: '{key}' not in {args.dataset} meta/stats.json")
    return torch.tensor(np.asarray(stats[key][field], dtype=np.float32))

added = {}
for key in ("observation.state", "observation.images.top", "observation.images.wrist"):
    buf = "normalize_inputs.buffer_" + key.replace(".", "_")
    added[buf + ".mean"] = t(key, "mean")
    added[buf + ".std"] = t(key, "std")
for prefix in ("normalize_targets", "unnormalize_outputs"):
    added[f"{prefix}.buffer_action.mean"] = t("action", "mean")
    added[f"{prefix}.buffer_action.std"] = t("action", "std")

new = [k for k in added if k not in sd]
sd.update(added)
save_file(sd, sd_path, metadata={"format": "pt"})
print(f"[fix] added {len(new)} tensors to model.safetensors:")
for k in new:
    print("   ", k, tuple(added[k].shape))

if args.no_upload:
    print(f"[fix] patched local copy only: {model_dir}")
else:
    upload_folder(repo_id=args.model, folder_path=model_dir, repo_type="model",
                  commit_message="Add normalization stats for LeRobot 0.3.x inference")
    print(f"[fix] uploaded to https://huggingface.co/{args.model} — now run section 6 again")
