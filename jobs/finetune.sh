#!/bin/bash
#SBATCH -A a135                          # your project account
#SBATCH --job-name=qwen2vl_cs_long          # job name
#SBATCH --nodes=16                        # total nodes
#SBATCH --ntasks-per-node=1              # one launcher per node
#SBATCH --gpus-per-task=4               # GPUs per node
#SBATCH --time=01:30:00                  # walltime
#SBATCH --partition=normal               # partition
#SBATCH --output=job_outputs/%x.txt      # STDOUT → job_outputs/JOBNAME.txt

# ─────────── ensure output dir exists ───────────
mkdir -p job_outputs

# ─────────── cluster‐wide env ───────────────────
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib0
export HF_HOME=$SCRATCH/huggingface_home

# ───────── repo setup & cd ─────────────────────────
export REPO_HOME=$SCRATCH/code/LLaMA-Factory
cd $REPO_HOME

# ────────── launch distributed training ────────────
srun --label --export=ALL --environment=llama-factory bash -c '
  set -euo pipefail

  cd $REPO_HOME

  # Figure out this node’s rank by hostname lookup
  host_list=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
  rank=0
  for h in "${host_list[@]}"; do
    if [[ "$h" == "$(hostname)" ]]; then
      break
    fi
    rank=$((rank+1))
  done
  export NODE_RANK=$rank
  export NNODES=$SLURM_NNODES
  export FORCE_TORCHRUN=1

  echo "➤ Node $NODE_RANK / $NNODES starting training…"

  llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft.yaml
'

