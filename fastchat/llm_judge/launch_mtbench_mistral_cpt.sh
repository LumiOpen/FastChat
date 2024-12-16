#!/bin/bash
#SBATCH --job-name=mtbench1_mistral_cpt  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing


export PYTHONPATH="/scratch/project_462000444/zosaelai2"
export TRANSFORMERS_CACHE="/scratch/project_462000444/cache"

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000444/zosaelai2/.fastchat_venv/bin/activate

echo "$(python -c 'import torch; print(torch.cuda.is_available())')"
# produce model responses to MTBench questions
# syntax: 
# python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]

# generate responses from Mistral
python gen_model_answer.py \
        --model-path /scratch/project_462000353/converted-checkpoints/mistral-cpt/mistral_7B_iter_0011920_bfloat16 \
        --model-id finnish-mistral-7b-base \
        --num-gpus-total 8 \
        --num-gpus-per-model 8 \
        --lang fi


# grade model responses using GPT-4
# syntax:
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
# python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]

# python gen_judgment.py \
#         --model-list  swedish-mistral-7b-sft-eng-swe \
#         --parallel 2 \

# run pairwise comparison
python gen_judgment.py \
        --mode pairwise-baseline \
        --model-list finnish-mistral-7b-base \
        --baseline-model mistral-7b-v0.2-finnish \
        --judge-model gpt-4-turbo \
        --parallel 2 \
        --lang fi \

# show pairwise comparison results
# python show_result.py --mode pairwise-all --model-list poro-mixed-poro-full-extended poro-mixed-poro-full

python show_result.py \
                --mode pairwise-baseline \
                --model-list finnish-mistral-7b-base \
                --baseline-model mistral-7b-v0.2-finnish \
                --judge-model gpt-4-turbo \
                --lang fi \

# plot results in spider plot
# python plot_results.py

# echo "$(python -c 'import torch; print(torch.cuda.is_available())')"
