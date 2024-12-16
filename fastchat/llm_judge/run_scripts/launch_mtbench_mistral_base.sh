#!/bin/bash
#SBATCH --job-name=mtbench_mistral_base  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000444/zosaelai2/.fastchat_venv/bin/activate

# echo "$(python -c 'import torch; print(torch.cuda.is_available())')"
# produce model responses to MTBench questions
# syntax: 
# python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]

# generate responses from Mistral Base
python gen_model_answer.py \
        --model-path /scratch/project_462000353/models/Mistral-7B-v0.2 \
        --model-id mistral-7b-v0.2-finnish \
        --num-gpus-total 8 \
        --num-gpus-per-model 8 \
        --lang fi


