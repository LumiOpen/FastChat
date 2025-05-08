#!/bin/bash
#SBATCH --job-name=mtb_judging  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=small       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=16     # Number of cores (threads)
#SBATCH --time=01:00:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462000615  # Project for billing

export PYTHONPATH="/scratch/project_462000353/zosaelai2"
export HF_HOME="/scratch/project_462000353/hf_cache"

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000353/zosaelai2/.fastchat_venv/bin/activate


export OPENAI_API_KEY=xxxx

for trg in en fi  
do
        echo "##### Judging ${trg^^} #####"
        python gen_judgment.py \
                --model-list llama-3.3-70b-instruct-$trg \
                --parallel 2 \
                --lang $trg \
                --judge-model gpt-4o-2024-08-06

        # show results scores
        python show_result.py \
                --model-list llama-3.3-70b-instruct-$trg \
                --lang $trg \
                --judge-model gpt-4o-2024-08-06

done
