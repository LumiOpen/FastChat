#!/bin/bash
#SBATCH --job-name=env_setup  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
##SBATCH --mem=0
##SBATCH --exclusive=user
##SBATCH --hint=nomultithread
#SBATCH --time=00:10:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000615  # Project for billing

# Set up virtual environment for Megatron-DeepSpeed pretrain_gpt.py.

# This script creates the directories venv and apex. If either of
# these exists, ask to delete.

mkdir -p logs
# Load modules

module load LUMI
module use /appl/local/csc/modulefiles/
module load pytorch


# Create and activate venv
python -m venv --system-site-packages .fastchat_venv
source .fastchat_venv/bin/activate
pip install --upgrade pip setuptools
git clone https://github.com/LumiOpen/FastChat.git
cd FastChat && pip install -e ".[model_worker,llm_judge]"

