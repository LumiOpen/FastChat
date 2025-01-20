#!/bin/bash
#SBATCH --job-name=mtbench1_mistral_cpt  # Job name
#SBATCH --output=.log/%j.out # Name of stdout output file
#SBATCH --error=.log/%j.err  # Name of stderr error file
#SBATCH --partition=small-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=3:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing
# Set up virtual environment for Megatron-DeepSpeed pretrain_gpt.py.

# This script creates the directories venv and apex. If either of
# these exists, ask to delete.

#wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

mkdir -p .log
# for the model judgements
mkdir -p data/mt_bench/model_judgment/
mkdir -p data/mt_bench/plots

# Load modules

module load LUMI
module use /appl/local/csc/modulefiles/
module load pytorch

# Create and activate venv
#python -m venv --system-site-packages .fastchat_venv
# install fastchat module and anthropic and fasttext 
source .fastchat_venv/bin/activate


export HF_HOME=/scratch/project_462000444/cache
export TRANSFORMERS_CACHE=/scratch/project_462000444/cache
export OPENAI_API_KEY=

lang=da
model_id=viking-33b-instruction-collection-packed-epochs-3-${lang}
eval_mode=single
judge_model=gpt-4

# produce model responses to MTBench questions
# syntax: 
# python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]

# Took 03:09:19 for the first part
python gen_model_answer.py \
       --model-path /scratch/project_462000444/zosaelai2/models/viking-33b-instruction-collection-packed-epochs-3 \
        --model-id $model_id \
        --num-gpus-total 8 \
        --num-gpus-per-model 8 \
        --lang $lang

# grade model responses using GPT-4
python gen_judgment.py \
        --model-list ${model_id}  \
        --mode single \
        --parallel 2 \
        --judge-model ${judge_model} \
        --lang ${lang}

# show pairwise comparison results
# python show_result.py -mode-mode pairwise-all --model-list poro-mixed-poro-full-extended poro-mixed-poro-full

python show_result.py \
        --mode single \
        --model-list ${model_id} \
        --judge-model ${judge_model} \
        --lang ${lang} \

# plot results in spider plot
python plot_results.py \
        --judgment-file data/mt_bench/model_judgment/${judge_model}_${eval_mode}.jsonl \
        --lang ${lang}

