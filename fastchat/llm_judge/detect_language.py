import fasttext
import argparse
from huggingface_hub import hf_hub_download
import os
import pandas as pd
from tqdm import tqdm
from heliport import Identifier
import json

tqdm.pandas()

def detect_language(text):
    """Given a text, it returns the prediction as NLLB language code, e.g., Latn-eng
    """
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = fasttext.load_model(model_path)

    lab, score = model.predict(text)
    first = lab[0]
    return first

def detect_language_heli(text):
    i = Identifier()
    three_ltr_code = i.identify(text)
    return three_ltr_code

def load_iso2nllb_map(filepath, mode):
    lang_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                two_letter_code, long_code = line.strip().split()
                if mode == 'iso2nllb':
                    lang_dict[two_letter_code] = long_code
                elif mode == 'nllb2iso':
                    lang_dict[long_code] = two_letter_code
    return lang_dict

def map_lang_code(lang_code, mapdict):
    return mapdict.get(lang_code, "Unknown code")

three2two = {
    "bul": "bg",  # Bulgarian
    "hrv": "hr",  # Croatian
    "ces": "cs",  # Czech
    "nld": "nl",  # Dutch
    "est": "et",  # Estonian
    "fra": "fr",  # French
    "deu": "de",  # German
    "ell": "el",  # Greek
    "hun": "hu",  # Hungarian
    "isl": "is",  # Icelandic
    "gle": "ga",  # Irish
    "ita": "it",  # Italian
    "lav": "lv",  # Latvian
    "lit": "lt",  # Lithuanian
    "mlt": "mt",  # Maltese
    "nno": "nn",  # Norwegian Nynorsk
    "pol": "pl",  # Polish
    "por": "pt",  # Portuguese
    "ron": "ro",  # Romanian
    "slk": "sk",  # Slovak
    "slv": "sl",  # Slovenian
    "spa": "es",  # Spanish
    "fin": "fi",  # Finnish
    "dan": "da",  # Danish
    "nor": "no",  # Norwegian
    "swe": "sv",   # Swedish
    "eng": "en",   # Swedish
    "hbs": "hr", # Serbocroatian accecpted as croatian
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect language of the input text file.")
    parser.add_argument("--input-file", type=str, help="Path to the input text file.")
    parser.add_argument("--target-language", type=str, help="Target language to compare against if not in filename after '_'")
    args = parser.parse_args()

    # Get the filename from the input file path
    filename = os.path.basename(args.input_file)

    # get mapping between long and two-letter lang codes
    path_to_langmap = "/scratch/project_462000353/maribarr/translation_scripts/iso2nllb.map"
    iso2nllb_dict = load_iso2nllb_map(path_to_langmap, mode="iso2nllb")
    nllb2iso_dict = load_iso2nllb_map(path_to_langmap, mode="nllb2iso")
    
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
    print(f"Target language: {tgt_lang}")
    tgt_lang_long = iso2nllb_dict[tgt_lang]

    df = pd.read_json(args.input_file, lines=True)

    # Ensure that 'turns' is a column
    df['turns'] = df['choices'].apply(lambda x: x[0]['turns'])
    df['category'] = df.question_id.map(lambda x: question_category_dict[x])

    # take out math and coding because language detection is less relevant here
    df = df.loc[~df.category.isin(['math', 'coding'])]

    #explode to get a list item per line
    df= df.explode(['turns']).reset_index(drop=True)

    #replace newlines - otherwise language detection fails
    df.loc[:, 'turns'] = df.turns.str.replace('\n', ' ')

    #detect language
    tqdm.pandas(desc="Detecting languages")
    df['pred_lang_long'] = df.turns.progress_map(lambda x: detect_language(text=x))
    df['pred_lang'] = df.pred_lang_long.map(lambda x: nllb2iso_dict.get(x[9:]))

    df['pred_lang_heli_long'] = df.turns.progress_map(lambda x: detect_language_heli(text=x))
    df['pred_lang_heli'] = df.pred_lang_heli_long.map(lambda x: three2two.get(x))

    # get acc exact match of top prediction
    acc = len(df[df.pred_lang==tgt_lang])/len(df)
    acc_heli = len(df[df.pred_lang_heli==tgt_lang])/len(df)

    print(f"Acc with Fasttext from {len(df)} datapoints for {tgt_lang}: {acc*100}")
    print(f"Acc with Heli from {len(df)} datapoints for {tgt_lang}: {acc_heli*100}")




