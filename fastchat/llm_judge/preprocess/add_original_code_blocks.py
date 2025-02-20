import pandas as pd
import glob
import re
import os

"""
This script reads the English MT Bench question file and a translated MT bench question file ``` ```
It puts the orignal code block in the correct turn or reference in all translated mt bench files 
"""

def extract_code(text:str)->str:
    """A function that detects code in markdown ``` ```and returns the code"""
    # Regular expression to match text enclosed in triple backticks
    pattern = r'```(.*?)```'
    
    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Split the text into parts, including the text blocks
    parts = re.split(pattern, text, flags=re.DOTALL)
    
    # Separate the surrounding text and the text blocks
    surrounding_texts = parts[::2]
    text_blocks = parts[1::2]
    
    return text_blocks, surrounding_texts

    
# Define the compare_and_fix function
def compare_and_fix(orig_text: str, translated_text: str) -> str:
    orig_code, _ = extract_code(orig_text)
    _, translated_text_parts = extract_code(translated_text)

    # Combine the translated text and original code blocks
    fixed_translation = ''.join(translated_text_parts)
    fixed_translation += ''.join(f"```{code}```" for code in orig_code)
    
    return fixed_translation

# Define the apply_compare_and_fix function
def apply_compare_and_fix(row, col_name):
    if isinstance(row[f'{col_name}_orig'], list) and isinstance(row[col_name], list):
        return [
            compare_and_fix(row[f'{col_name}_orig'][0], row[col_name][0]) if row[f'{col_name}_orig'][0] and row[col_name][0] else None,
            compare_and_fix(row[f'{col_name}_orig'][1], row[col_name][1]) if row[f'{col_name}_orig'][1] and row[col_name][1] else None
        ]
    else:
        return row[f'{col_name}_orig']


if __name__=='__main__':
    #input_path = "/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/gemini"
    input_path = 'data/mt_bench'
    df_eng = pd.read_json('data/mt_bench/question.jsonl', lines=True)

    for input_fname in glob.glob(input_path+'/*.jsonl'):
        fbase_name = os.path.basename(input_fname)
        if fbase_name not in ['questions.jsonl' or 'en_questions.jsonl'] and 'question_' in fbase_name:
            print(f'Processing {input_fname}')

            df = pd.read_json(input_fname, lines=True)
            output_cols = df.columns

            df_merged = df.merge(df_eng[['question_id', 'reference', 'turns' ]], on='question_id', how='left', suffixes=['', '_orig'])
            # Apply the compare_and_fix function to each pair of items in the 'turns' and 'turns_orig' columns

            df_merged['fixed_turns'] = df_merged.apply(lambda row: apply_compare_and_fix(row, 'turns'), axis=1)

            df_merged['fixed_reference'] = df_merged.apply(lambda row: apply_compare_and_fix(row, 'reference'), axis=1)

            # overwrite turns and reference
            df_merged['turns'] = df_merged.fixed_turns
            df_merged['reference'] = df_merged.fixed_reference

            #df_merged.loc[df_merged.question_id==124]['turns'].map(lambda x: print(x[0]))

            #remove extra new cols
            df_merged = df_merged[output_cols]
            
            df_merged.to_json(input_fname, orient='records', lines=True)
