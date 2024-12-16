#!/bin/bash
#SBATCH --job-name=mtbench_vik_fi  # Job name
#SBATCH --account=project_462000353  # Project for billing
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00       # Run time (d-hh:mm:ss)


module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000444/zosaelai2/.fastchat_venv/bin/activate

echo "$(python -c 'import torch; print(torch.cuda.is_available())')"
# produce model responses to MTBench questions
# syntax: 
# python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]

python gen_model_answer.py \
        --model-path /scratch/project_462000444/zosaelai2/models/viking-33b-instruction-collection-packed-epochs-3 \
        --model-id viking-33b-instruction-collection-packed-epochs-3 \
        --num-gpus-total 8 \
        --num-gpus-per-model 8 \
        --lang fi

# grade model responses using GPT-4
# syntax:
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
# python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]

python gen_judgment.py \
        --model-list viking-33b-instruction-collection-packed-epochs-3  \
        --parallel 2 \
        --lang fi


# show results scores
python show_result.py --model-list viking-33b-instruction-collection-packed-epochs-3 --lang fi

                
# plot results in spider plot
# python plot_results.py

# echo "$(python -c 'import torch; print(torch.cuda.is_available())')"
