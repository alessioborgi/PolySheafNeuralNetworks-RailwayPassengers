#!/bin/bash
#SBATCH --job-name=polynomial_neural_sheaf_diffusion_ablation
#SBATCH --partition=queue_dip_ingegneria
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/polynsd_weep_sweep_%j.out
#SBATCH --error=logs/polynsd_sweep_%j.err
#SBATCH --account=diaghpc

# Make sure logs directory exists
mkdir -p logs

# Load your environment (miniconda + myenv via .bashrc)
#source ~/.bashrc
CONDA_ROOT_PATH="/cm/shared/apps/linux-ubuntu22.04-zen2/miniconda3/24.3.0/4ycfox6czb6abkkr2x2bvs6gsmwnlqnn"

# 2. EXPLICITLY initialize Conda for the current non-interactive shell
source "$CONDA_ROOT_PATH/etc/profile.d/conda.sh"

# 3. ACTIVATE the environment where 'wandb' is installed (likely 'base')
conda activate esnn

echo "===== ENV CHECK ====="
echo "Host: $(hostname)"
echo "Python: $(which python)"
python - << 'EOF'
import sys, torch
print("Python:", sys.executable)
print("Torch:", torch.__file__)
EOF
echo "====================="

# Go to project directory
#cd Polynomial-Sheaf-Diffusion/

# Optional: ensure a dedicated HF cache on beegfs (overrides script default if you like)
# export HF_HOME=/mnt/beegfs/home/fwani/.hf_cache

echo ">>> Starting Polynomial Sheaf Diffusion Sweeping run..."

wandb agent adavit/ChebyshevInterpolationSheafsBundle_PubMed/qrzdq54e

echo ">>> Job finished."
