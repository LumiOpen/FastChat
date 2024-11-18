import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse

CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]



def get_model_df(judgment_file):
    cnt = 0
    q2result = []
    fin = open(judgment_file, "r")
    for line in fin:
        obj = json.loads(line)
        obj["category"] = CATEGORIES[(obj["question_id"]-81)//10]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment-file", type=str, default="data/mt_bench/model_judgment/gpt-4_single.jsonl")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    if args.lang != "en":
        judgment_file = args.judgment_file.replace(".jsonl", f"_{args.lang}.jsonl")
    else:
        judgment_file = args.judgment_file
    print("judgment_file:", judgment_file)
    df = get_model_df(judgment_file)
    all_models = df["model"].unique()
    # print(all_models)
    scores_all = []
    for model in all_models:
        for cat in CATEGORIES:
            # filter category/model, and score format error (<1% case)
            res = df[(df["category"]==cat) & (df["model"]==model) & (df["score"] >= 0)]
            score = res["score"].mean()

            # # pairwise result
            # res_pair = df_pair[(df_pair["category"]==cat) & (df_pair["model"]==model)]["result"].value_counts()
            # wincnt = res_pair["win"] if "win" in res_pair.index else 0
            # tiecnt = res_pair["tie"] if "tie" in res_pair.index else 0
            # winrate = wincnt/res_pair.sum()
            # winrate_adjusted = (wincnt + tiecnt)/res_pair.sum()
            # # print(winrate_adjusted)

            # scores_all.append({"model": model, "category": cat, "score": score, "winrate": winrate, "wtrate": winrate_adjusted})
            scores_all.append({"model": model, "category": cat, "score": score})

    target_models = [

    ]

    scores_target = [scores_all[i] for i in range(len(scores_all)) if scores_all[i]["model"] in target_models]
    # sort by target_models
    scores_target = sorted(scores_target, key=lambda x: target_models.index(x["model"]), reverse=True)

    df_score = pd.DataFrame(scores_target)
    df_score = df_score[df_score["model"].isin(target_models)]


    rename_map = {

    }

    for k, v in rename_map.items():
        df_score.replace(k, v, inplace=True)

    # plotting results
    fig = px.line_polar(df_score, r = 'score', theta = 'category', line_close = True, category_orders = {"category": CATEGORIES},
                        color = 'model', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel,
                        title=f"MTBench {args.lang.upper()}",)

    fig.update_layout(
        font=dict(
            size=15,
        ),
    )

    # fig.write_image("fig.png", width=900, height=600, scale=1)
    fig.write_html(f"mtbench_{args.lang}_poro_argilla.html")
