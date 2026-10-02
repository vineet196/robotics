#!/bin/bash
# =====================================================================
#  Group HW2 — ACT training job for The Mill (CS 5403, Missouri S&T)
#  v4 (Sep 29): reads the dataset converted in handout section 5.0 (--dataset.root)
#  v3 (Sep 22): shared env path; rss-class partition; GPU via --gres=shard:1
#  Edit ONLY the two lines marked  <-- EDIT.  Leave everything else.
#  Submit with:   sbatch train_act.sbatch
# =====================================================================
#SBATCH --job-name=act_pickplace
#SBATCH --partition=rss-class
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-03:30:00
#SBATCH --output=mill-%j.out
#SBATCH --error=mill-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=YOUR_SSO@umsystem.edu          # <-- EDIT (1)

# ---- team settings -------------------------------------------------
HF_ID=YOUR_TEAM_LEAD_HF_ID                          # <-- EDIT (2)
DATASET=$HF_ID/so101_pickplace_v1                   # must match your lerobot-record repo_id
MODEL=$HF_ID/act_so101_pickplace_v1
DATASET_DIR=/share/ceph/scratch/$USER/so101_pickplace_v1   # converted copy from section 5.0

# ---- environment (do not change) -----------------------------------
source /share/apps/common/anaconda/2023.09/etc/profile.d/conda.sh
conda activate /share/apps/common/conda-envs-global/lerobot312
# Reminder: run `hf auth login` once on the login node before submitting,
# so the job can push your trained model to the Hub.

echo "### Group HW2 ACT training — start $(date) ###"
echo "dataset=$DATASET  model=$MODEL  HF_HOME=$HF_HOME"
if ! grep -q '"codebase_version": "v3.0"' "$DATASET_DIR/meta/info.json" 2>/dev/null; then
  echo "ERROR: $DATASET_DIR is missing or not converted to v3.0 - do handout section 5.0 first."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

lerobot-train \
  --dataset.repo_id=$DATASET \
  --dataset.root=$DATASET_DIR \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=$MODEL \
  --output_dir=/share/ceph/scratch/$USER/outputs/act_v1_$SLURM_JOB_ID \
  --job_name=act_v1 \
  --batch_size=8 \
  --steps=40000 \
  --log_freq=200 \
  --save_freq=10000 \
  --wandb.enable=false

echo "### end $(date) ###"
