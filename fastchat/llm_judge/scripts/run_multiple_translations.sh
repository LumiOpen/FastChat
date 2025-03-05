#!/bin/bash
#SBATCH --job-name=multitranslations  # Job name
#SBATCH --output=.log/%j.out # Name of stdout output file
#SBATCH --error=.log/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=40:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing

#python3 -m fastchat.serve.cli --model /scratch/project_462000444/zosaelai2/models/europa_90pct_sft --debug

mkdir -p .log
# for the model judgements

export PYTHONPATH="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/.fastchat_venv/lib/python3.10/site-packages"
export HF_HOME="/scratch/project_462000444/cache"


MODEL_PATH=/scratch/project_462000444/zosaelai2/models/europa_90pct_sft
MODEL_PATH=/scratch/project_462000444/zosaelai2/models/europa_100pct_sft_data_mix_v0.1
BASE_MODEL_PATH=/scratch/project_462000353/converted-checkpoints/europa_7B_iter_0715255_bfloat16
MODEL_NAME="europa_100pct_sft"
BENCH_NAME="multi_translation"
PROMPT_PATH="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/multi_translation/base_model_prompts/multilingual_3.txt"
#PROMPT_PATH="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/multi_translation/base_model_prompts/multilingual_2.txt"
NUM_CHOICES=100
SYSTEM_MESSAGE="You are a helpful assistant translating English sentences into other languages."
INSTRUCTION="Translate the following English sentence into {arg1}: "

mkdir -p data/${BENCH_NAME}/model_judgment/
mkdir -p data/${BENCH_NAME}/plots
mkdir -p data/${BENCH_NAME}/model_answer/${MODEL_NAME}

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/.fastchat_venv/bin/activate

echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

#LANGUAGES=("bg" "hr" "cs" "nl" "et" "fr" "de" "el" "hu" "is" "ga" "it" "lv" "lt" "mt" "nn" "pl" "pt" "ro" "sk" "sl" "es" "fi" "da" "no" "sv")
#LANGUAGES=("bg" "hr" "cs" "nl" "et" "fr" "de" "el" "hu" "is" "ga" "it" "lv" "lt" "mt" "nn" "pl" "pt" "ro" "sk" "sl" "es" "fi" "da" "sv")
LANGUAGES=("da" "sv" "es" "fr" "pt" "de" "no" "nn")

for lang in "${LANGUAGES[@]}"; do
        #python multiple_translation.py \
        #       --model-path $MODEL_PATH \
        #        --model-id  $MODEL_NAME \
        #       --num-gpus-total 8 \
        #       --num-gpus-per-model 8 \
        #       --lang $lang \
        #       --bench-name $BENCH_NAME \
        #       --num-choices $NUM_CHOICES \
        #       --system-message "$SYSTEM_MESSAGE" \
        #       --instruction-template "$INSTRUCTION" \
        
        #python detect_language.py --input-file data/${BENCH_NAME}/model_answer/${MODEL_NAME}/answers_${lang}.jsonl
        python query_base_model.py \
        --model-path $BASE_MODEL_PATH \
        --input-file data/${BENCH_NAME}/model_judgement/${MODEL_NAME}/translations_${lang}.jsonl \
        --prompt $PROMPT_PATH
done

