import argparse
import os
import pandas as pd
from tqdm import tqdm
import json
from common import get_lang_code_dict, detect_language_fasttext, detect_language_glotlid

tqdm.pandas()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect language of the input text file.")
    parser.add_argument("--input-file", type=str, help="Path to the input text file.")
    parser.add_argument("--target-language", type=str, help="Target language to compare against if not in filename after '_'")
    args = parser.parse_args()

    # Get the filename from the input file path
    filename = os.path.basename(args.input_file)
    
    # Read the JSONL file line by line and load each JSON object individually
    data = []
    file_path = 'data/mt_bench/question.jsonl'
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    # Convert the list of JSON objects to a DataFrame
    df_eng = pd.DataFrame(data)

    # Create a dictionary with question_id as the key and category as the value
    question_category_dict = df_eng.set_index('question_id')['category'].to_dict()

    # If target_language is not provided, extract it from the filename
    if not args.target_language:
        tgt_lang = filename.split('_')[-1].split('.')[0]
        if tgt_lang == 'no':
            tgt_lang = 'nb'
    else:
        tgt_lang = args.target_language

    model_name = args.input_file.split('/')[-2]

    lang_dict_tgt_lang = get_lang_code_dict(tgt_lang)
    # keys for this dict is alpha3-b,alpha3-t,alpha2,English,French
    tgt_lang_eng = lang_dict_tgt_lang['English']
    tgt_lang_3_ltr = lang_dict_tgt_lang['alpha3-b']
    if tgt_lang_3_ltr == None or tgt_lang_eng == None:
        print('Language not found')
    else:
        print(f"Target language: {tgt_lang}, {tgt_lang_eng}, {tgt_lang_3_ltr}")
    
    # english does not use the suffix
    if tgt_lang == 'en':
        input_file = args.input_file.str.replace('_en', '')
    else:
        input_file = args.input_file
    df = pd.read_json(input_file, lines=True)

    # Ensure that 'turns' is a column
    df['turns'] = df['choices'].apply(lambda x: x[0]['turns'])
    df['category'] = df.question_id.map(lambda x: question_category_dict[x])

    # take out math and coding because language detection is less relevant here
    # Consider taking out extraction because it request json
    df = df.loc[~df.category.isin(['math', 'coding', 'extraction'])]

    #explode to get a list item per line
    df= df.explode(['turns']).reset_index(drop=True)

    #replace newlines - otherwise language detection fails
    df.loc[:, 'turns'] = df.turns.str.replace('\n', ' ')

    #detect language
    tqdm.pandas(desc=f"Detecting languages for {tgt_lang_eng} with FastText")
    df['pred_lang_ft_long'] = df.turns.progress_map(lambda x: detect_language_fasttext(text=x))
    
    df['pred_lang_ft'] = df.pred_lang_ft_long.map(lambda x: get_lang_code_dict(x.split('_')[-2]).get('alpha2'))

    tqdm.pandas(desc=f"Detecting languages for {tgt_lang_eng} with Glotlid")
    df['pred_lang_gl_long'] = df.turns.progress_map(lambda x: detect_language_glotlid(text=x))
    df['pred_lang_gl'] = df.pred_lang_gl_long.map(lambda x: get_lang_code_dict(x.split('_')[-2]).get('alpha2'))

    # get acc exact match of top prediction
    acc_ft = len(df[df.pred_lang_ft==tgt_lang])/len(df)
    acc_gl = len(df[df.pred_lang_gl==tgt_lang])/len(df)

    print(f"Acc with Fasttext from {len(df)} datapoints for {tgt_lang}: {acc_ft*100}")
    print(f"Acc with Glotlid from {len(df)} datapoints for {tgt_lang}: {acc_gl*100}")

    gl_errors = df[df.pred_lang_gl!=tgt_lang]
    ft_errors = df[df.pred_lang_ft!=tgt_lang]

    gl_errors.to_csv(f'data/mt_bench/errors/{model_name}/gl_{tgt_lang}.csv', index=None)
    ft_errors.to_csv(f'data/mt_bench/errors/{model_name}/ft_{tgt_lang}.csv', index=None)

    #print(df[df.pred_lang_gl!=tgt_lang][['pred_lang_gl', 'pred_lang_gl_long', 'turns', 'category']])
    #print(df[df.pred_lang_ft!=tgt_lang][['pred_lang_ft', 'pred_lang_ft_long', 'turns', 'category']])



