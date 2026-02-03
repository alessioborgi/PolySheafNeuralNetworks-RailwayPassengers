#!/bin/bash
#SBATCH -J PolySheaf_Sweep
#SBATCH --account=LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/ab3352/Polynomial-Neural-Sheaf-Diffusion/logs/%x_%j.out
#SBATCH --error=/home/ab3352/Polynomial-Neural-Sheaf-Diffusion/logs/%x_%j.err

echo "=== Job Start: $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

REPO_DIR=/home/ab3352/Polynomial-Neural-Sheaf-Diffusion
mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR" || exit 1

# ---- Conda (non-interactive) ----
CONDA_BASE=/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh
source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "ERROR: cannot source conda.sh"; exit 1; }
conda activate poly-nsd-mac || { echo "ERROR: cannot activate env poly-nsd-mac"; exit 1; }

echo "Python: $(which python)"
echo "Wandb:  $(which wandb || echo NOT_FOUND)"
wandb --version || { echo "ERROR: wandb not available in this env"; exit 1; }

wandb login
echo "Launching W&B Agent..."
# wandb agent adavit/ChebyshevT1SheafsDiagNSD_AmazonRatings/iuxhaxbj
# wandb agent adavit/ChebyshevT1SheafsDiag_AmazonRatings/s2xo2y9s

# wandb agent adavit/ChebyshevT1SheafsDiagNSD_RomanEmpire/fjqmcltf
# wandb agent adavit/ChebyshevT1SheafsDiag_RomanEmpire/4a4u8ldq

# wandb agent adavit/ChebyshevT1SheafsDiagNSD_Ogbn_Arxiv/u5p18jmy
# wandb agent adavit/ChebyshevT1SheafsDiag_Ogbn_Arxiv/37k5s73a

# wandb agent adavit/ChebyshevT1SheafsDiagNSD_WikiCS/x1363pjf
# wandb agent adavit/ChebyshevT1SheafsDiag_WikiCS/jqhns698

wandb agent adavit/CityNetwork_Influence_RF/cya5773o

echo "=== Job End: $(date) ==="
