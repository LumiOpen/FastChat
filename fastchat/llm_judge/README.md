## Original MTBench paper and leaderboard
| [Paper](https://arxiv.org/abs/2306.05685) | [Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) |

## Finnish MT-Bench
We extend the original MTBench to evaluate chat models in Finnish. We plan to support additional Nordic languages in the near future. 
We translate the questions and reference answers from MTBench into Finnish using [DeepL](https://www.deepl.com/translator). 

We added language identification when judging the model answers so that the language of the answer should match the language of the question. If there is a language mismatch, we automatically set the score to [[1]] for single-answer grading without calling the LLM judge. 
For pairwise judging, if the language of Model A does not match the language of the question but Model B does then B automatically wins. If both match, the pair of answers is passed to the judge. If neither model matches, the verdict is automatically an error.

## Contents
- [Install](#install)
- [MT-Bench](#mt-bench)


## Install
```
git clone https://github.com/LumiOpen/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
pip install fasttext
```
Download [FastText language identifier](https://fasttext.cc/docs/en/language-identification.html) and change `FASTTEXT_LID_BINARY` in [common.py](https://github.com/LumiOpen/FastChat/blob/main/fastchat/llm_judge/common.py) to point to your local lid binary.

## MT-Bench

### Evaluate a model on MT-Bench

#### Step 1. Generate model answers to MT-Bench questions
```
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID] --lang [LANG-CODE]
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[MODEL-ID]` is a name you give to the model.
  - `[LANG-CODE]` two-letter language code. Choices are `en` or `fi`. Default is `en`.

e.g. 

**English**
```
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --num-gpus-total 8 --num-gpus-per-model 8 
```
**Finnish**
```
python gen_model_answer.py --model-path LumiOpen/Poro-34B-Chat-beta --model-id Poro-34B-Chat-beta --num-gpus-total 8 --num-gpus-per-model 8  --lang fi
```

The answers will be saved to `data/mt_bench/model_answer/[MODEL-ID].jsonl` regardless of the language.

To make sure FastChat loads the correct prompt template, see the supported models and how to add a new model [here](../../docs/model_support.md#how-to-support-a-new-model).

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommend single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

For Finnish, indicate the language code `fi` so that it uses the Finnish questions and reference answers for grading. The judge prompt to GPT-4 is the same for all languages. 

We include language checking at this stage so that if the primary language of the answer does not match the primary language of the question, the answer is automatically given a score of 1 (lowest possible score) and skip querying GPT-4. 
```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --lang [LANG-CODE]
```

e.g.

**English**
```
python gen_judgment.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4 --parallel 2
```
The judgments for English will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

**Finnish**
```
python gen_judgment.py --model-list Poro-34B-Chat-beta --parallel 2 --lang fi
```
Finnish judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single_finnish.jsonl`



#### Step 3. Show MT-Bench scores

**Finnish**

- Show the scores for selected models
  ```
  python show_result.py --model-list Poro-34B-Chat-beta --lang fi
  ```
- Plot results as a spider plot
  ```
  python plot_results.py --judgment-file data/mt_bench/model_judgment/gpt-4_single_finnish.jsonl
  ```

---

### Other grading options 
Besides score-based single-answer grading, we also support two additional grading options based on win rates:
- `pariwise-baseline`: run pairwise comparison against a baseline model.
- `pairwise-all`: run pairwise comparison between all model pairs on all questions.

#### Option 2: pairwise comparison against a baseline (default: gpt-3.5-turbo)

- Generate GPT-4 judgments

**English** 
```
python gen_judgment.py --mode pairwise-baseline --model-list vicuna-13b-v1.3 alpaca-13b llama-13b --parallel 2
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_pair.jsonl`

**Finnish** 
```
python gen_judgment.py --mode pairwise-baseline --model-list Poro-34B-Chat-beta llama-7b-finnish-instruct-v0.2 --parallel 2 --lang fi
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_pair_finnish.jsonl`

- Show results
```
python show_result.py --mode pairwise-baseline --lang fi
```

#### Option 3: Run GPT-4 judge with all pair comparisons

Another option is to run pairwise comparisons on all possible pairs.
This could be more expensive when #models increases, but it gives you a more comprehensive information.

```
python gen_judgment.py --mode pairwise-all --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --lang [LANG-CODE]
```

```
python show_result.py --mode pairwise-all --lang fi
```

### How to get GPT-3.5/GPT-4/Claude's answer? (Not yet supported for Finnish)
- `python gen_api_answer.py --model [MODEL-NAME]` to generate GPT-3.5/4 and Claude's answers.


### How to plot the radar figure?

You can use this [colab notebook](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO) to plot the radar figure for MT-bench.

<img src="data/mt_bench/misc/radar.png" width="600" height="450">

Alternatively, run this script with the path to the judgment file:
```
python plot_results.py --judgment-file data/mt_bench/model_judgment/gpt-4_single_finnish.jsonl
```

### Limitations
While the original MTBench has shown 80% agreement between human and GPT-4 ratings, we have not conducted the same experiment on the Finnish MT-Bench. Additionally, the DeepL translations of the questions and reference answers have not been verified by a native speaker.

