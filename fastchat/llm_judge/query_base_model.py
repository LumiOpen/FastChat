import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import argparse
import pandas as pd
from tqdm.auto import tqdm
import codecs
import os
from collections import defaultdict
from common import detect_language_glotlid

tqdm.pandas()

# Define a custom stopping criterion that stops generation when a specific token sequence is generated.
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        """
        Args:
            stop_ids (List[int]): The list of token IDs that, when generated consecutively at the end,
                                  will trigger stopping.
        """
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check only if enough tokens have been generated.
        if input_ids.shape[-1] >= len(self.stop_ids):
            # Check if the last tokens match the stop sequence.
            if input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids:
                return True
        return False

def strip_answer(indices:list, stop_token:str) -> str:
    # Decode the generated tokens back to text.
    generated_text = tokenizer.decode(indices, skip_special_tokens=True)
    # Extract the answer by removing the prompt part.
    # Since our prompt ends with "A:", we extract the text that comes after.
    answer = generated_text[len(prompt):].strip()
    answer = answer.split(stop_token)[0]
    return answer

def extract_logprobs_for_tokens(output, tokens, tokenizer):
    """
    Extract the log probabilities for specific tokens in the generated sequence.
    
    Args:
        output: The output from the model.generate method.
        tokens: A list of tokens for which to extract log probabilities.
        
    Returns:
        A dictionary of log probabilities for the specified tokens.
    """
    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens]

    logprob_dict = defaultdict(list)
    for score in output.scores: # I wonder if I should accumulate the logprobs for each token for the lenght of scores. Right now the max_new_tokens is set to 1 so it is not necessary
        probs = torch.nn.functional.log_softmax(score, dim=-1)
        for token_id in token_ids:
            if token_id < probs.size(-1): # if token id is within the range of vocab size
                logprob_dict[tokenizer.decode([token_id])].append(probs[0, token_id].item())
    return logprob_dict

def get_ratings_from_prompt(prompt:str) -> list:
    """Get the the set of ratings that occur on the prompt file by taking everything after Rating: in a line"""
    t = prompt.split('\n')
    ratings = list(set([l.split(':')[-1].strip() for l in t if "Rating:" in l[:8]])) # Sensitive to the word 'Rating:' occurring as the first word in a line
    return ratings

def rate_translation(source_sentence: str, translation: str, model, tokenizer, few_shot_prompt: str) -> str:
    """
    Appends the given question to the few-shot prompt, generates an answer, and returns the answer text.
    
    Args:
        question (str): The question prompt to answer.
        
    Returns:
        str: The generated answer.
    """

    # Define the stop token string to be used after the few-shot prompt.
    few_shot_prompt_stop_tokens = "\n"

    # Build the complete prompt including the new question.
    prompt = f"{few_shot_prompt}\n\nSource sentence: {source_sentence}\n Translation: {translation}\n Rating:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Convert the stop string to token IDs.
    stop_ids = tokenizer.encode(few_shot_prompt_stop_tokens, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
    
    # Generate the answer using our custom stopping criteria.
    output = model.generate(
        input_ids,
        #max_length=256,  
        max_new_tokens=5,             # Adjust as needed. I wonder if I schould increase it
        do_sample=True,               # Enable sampling for more diverse outputs.
        temperature=0.8,              # Adjust randomness.
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
        output_scores=True,           # Output scores for logprobs
        return_dict_in_generate=True  # Return a dictionary with the outputs
    )
    
    #answer = strip_answer(output.sequences[0], stop_token=few_shot_prompt_stop_tokens)
    # read the set of unique ratings from the prompt file
    tokens = get_ratings_from_prompt(few_shot_prompt)
    
    logprob_dict = extract_logprobs_for_tokens(output, tokens, tokenizer=tokenizer)
    # Get the logprobs for the specified tokens

    max_logprob_token = max(logprob_dict, key=lambda k: max(logprob_dict[k]))

    return max_logprob_token


def pairwise_test_sentences(df, model, tokenizer, few_shot_prompt):
    results = []

    # Group the DataFrame by the original sentence
    grouped = df.groupby('original')

    for original, group in grouped:
        translations = group['translation'].tolist()

        for i, translation in enumerate(translations):
            # Select 5 random other candidates
            candidates = random.sample(translations[:i] + translations[i+1:], min(5, len(translations) - 1))

            for candidate in candidates:
                rating = rate_translation(source_sentence=original,
                                          translation=candidate,
                                          model=model,
                                          tokenizer=tokenizer,
                                          few_shot_prompt=few_shot_prompt)
                results.append({
                    'original': original,
                    'translation': translation,
                    'candidate': candidate,
                    'rating': rating
                })

# Example usage:
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        default = "/scratch/project_462000353/converted-checkpoints/europa_7B_iter_0715255_bfloat16",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
                        )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="The jsonl input file. Needs to have a column with original and translation",
                        )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        default='/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/multi_translation/base_model_prompts/multilingual_3.txt',
        help="Path to the basemodel few-shot prompt"
                                )
    
    args = parser.parse_args()

    # Load the tokenizer and model. Using GPT-2 for this example.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    """model, tokenizer = load_model(
        args.model_path,
        revision='main',
        device="cuda",
        num_gpus=8,
        max_gpu_memory=256,
        dtype='float16',
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )"""

    prompt = codecs.open(args.prompt, 'r', 'utf-8').read()
    prompt_fname = os.path.basename(args.prompt).strip('.txt')

    source_sentence = "Nobody in the beginning thought it was possible, and nobody wanted to help us"
    #ranslation = "Ingen tror på det umulige, og ingen ville hjælpe os" #2
    #translation = "Ingen ville tro det var muligt i begyndelsen, og ingen ville hjælpe os" # 1
    translation = "Ingen håbede det var muligt, og ingen ville hjælpe os" # 2
    #translation = "Ingen hjalp os" # 3
    #translation = "Nobody in the beginning thought it was possible, and nobody wanted to help us" #3
    answer = rate_translation(source_sentence=source_sentence,
                             translation=translation, 
                             model=model,
                             tokenizer=tokenizer,
                             few_shot_prompt=prompt)
    print("Source:", source_sentence)
    print('Translation: ', translation)
    print("Rating:", answer)

    # reading the file
    df = pd.read_json(args.input_file, lines=True)

    df[prompt_fname] = df.progress_apply(lambda x: rate_translation(source_sentence=x['original'],
                             translation=x['translation'], 
                             model=model,
                             tokenizer=tokenizer,
                             few_shot_prompt=prompt), axis=1)
    
    # save as the same filename - just with an extra column
    #df.to_json(args.input_file, lines=True, orient='records')
    #df.to_json(f"test.jsonl", lines=True, orient='records')


