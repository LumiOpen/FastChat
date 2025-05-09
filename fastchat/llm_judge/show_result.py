"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd
import json

def display_result_single_by_category(args):
    if args.lang != 'en':
        questions = [json.loads(line) for line in open(f"data/mt_bench/question_{args.lang}.jsonl")]
    else:
        questions = [json.loads(line) for line in open("data/mt_bench/question.jsonl")]
    categories = sorted(set([entry['category'] for entry in questions]))
    question_ids = {c: [] for c in categories}
    for category in categories:
        for entry in questions:
            if entry['category'] == category:
                question_ids[category].append(entry['question_id'])
    if args.input_file is None:
        if args.lang != 'en':
            input_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_single_{args.lang}.jsonl"
            )
        else:
            input_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
            )            
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn", "question_id"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]
    
    for category in categories:
        print("\n##########", category, "##########")
        df_1 = df[df["question_id"].isin(question_ids[category])].groupby(["model"]).mean()
        print(df_1.sort_values(by="score", ascending=False))

def display_result_single(args):
    if args.input_file is None:
        if args.lang != 'en':
            input_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_single_{args.lang}.jsonl"
            )
        else:
            input_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
            )            
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn", "judgment"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\n########## Language check ##########")
    print("Language mismatch:", df[df.judgment.str.contains("Language error")].shape[0])
    df = df[["model", "score", "turn"]]
    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))


def display_result_pairwise(args):
    if args.input_file is None:
        if args.lang != "en":
            input_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair_{args.lang}.jsonl"
            )
        else:
            input_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
            )
    else:
        input_file = args.input_file
    if args.lang == 'en':   
        questions = [json.loads(line) for line in open("data/mt_bench/question.jsonl")]
    else:
        questions = [json.loads(line) for line in open(f"data/mt_bench/question_{args.lang}.jsonl")]
    if args.exclude_category is not None and len(args.exclude_category) > 0:
        valid_question_ids = [line['question_id'] for line in questions if line['category'] not in args.exclude_category]
    else:
        valid_question_ids = [line['question_id'] for line in questions]

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    if valid_question_ids != None:
        df_all = df_all[df_all['question_id'].isin(valid_question_ids)]
    model_1 = args.model_list[0]
    if args.baseline_model is not None:
        model_2 = args.baseline_model
    else:
        model_2 = args.model_list[1]
    print("model_1:", model_1)
    print("model_2:", model_2) 
    df_all = df_all[(df_all["model_1"] == model_1) & (df_all["model_2"] == model_2)]
    df_all = df_all[(df_all["g1_judgment"] != "$ERROR$") | (df_all["g2_judgment"] != "$ERROR$")]
    # print("df_all:", df_all)
    df_errors = df_all[(df_all["g1_winner"] == "error") & (df_all["g2_winner"] == "error")]
    print("errors:", len(df_errors))

    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))

    list_res = []
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"] or row['g1_winner'] == 'error':
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    df = df.groupby(["model"]).sum()

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df.index != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    # each tie counts as 0.5 win + 0.5 loss
    df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
        df["win"] + df["loss"] + df["tie"]
    )
    # print(df.sort_values(by="win_rate", ascending=False))
    # print(df.sort_values(by="loss_rate", ascending=True))
    print(df.sort_values(by="win_rate_adjusted", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-2024-08-06", help="gpt-4o-2024-08-06, gpt-4o is the default judge")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument("--lang", type=str, default="en", help="supported langs: en, fi, sv, da, no, is")
    parser.add_argument(
        "--exclude-category",
        type=str,
        nargs="+",
        default=None,
        help="exclude some categories from the score summary",
    )
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)
