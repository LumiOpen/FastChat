#!/bin/bash
#SBATCH --job-name=mtbench_gpt  # Job name
#SBATCH --account=project_462000353  # Project for billing
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=03:00:00       # Run time (d-hh:mm:ss)

export PYTHONPATH="/scratch/project_462000444/zosaelai2"
export HF_HOME="/scratch/project_462000444/cache"

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000444/zosaelai2/.fastchat_venv/bin/activate

# produce model responses to MTBench questions
# syntax: 
# python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]

export OPENAI_API_KEY=xxxx
python gen_api_answer.py \
        --model gpt-4-turbo \
        --lang en \

