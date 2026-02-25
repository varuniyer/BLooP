import math
from collections import defaultdict
from typing import cast

import evaluate
import pandas as pd
import toml
from datasets import (
    Dataset,
    load_dataset,  # type: ignore
)
from nltk import sent_tokenize  # type: ignore
from tqdm import tqdm
from wandb.apis.public.api import Api
from wandb.sdk.wandb_run import Run

from bart_score import BARTScorer
from utils import get_df_from_wandb


def get_scorers() -> tuple[BARTScorer, evaluate.EvaluationModule]:
    bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
    rouge_scorer = evaluate.load("rouge")
    return bart_scorer, rouge_scorer


def get_datasets() -> dict[str, tuple[str, Dataset]]:
    dataset_info = cast(dict[str, dict[str, str]], toml.load("dataset_info.toml"))
    datasets: dict[str, tuple[str, Dataset]] = {}
    for ds_name, ds_info in dataset_info.items():
        ds_args = [ds_info["name"]]
        if ds_info["subset"]:
            ds_args.append(ds_info["subset"])
        datasets[ds_name] = (
            ds_info["ref_key"],
            cast(Dataset, load_dataset(*ds_args, split="test", trust_remote_code=True)),
        )
    return datasets


def collect_results() -> None:
    api = Api()
    datasets = get_datasets()
    bart_scorer, rouge_scorer = get_scorers()

    results_tables: dict[str, dict[float, pd.DataFrame]] = defaultdict(dict)
    perf_data: list[dict[str, float | int | str]] = []
    cache_usage_data: list[dict[str, float | int | str]] = []
    promotion_effect_data: list[dict[str, float | int | str]] = []
    for model in ["Llama", "Mistral", "Gemma"]:
        for run in tqdm(
            sorted(
                cast(
                    list[Run],
                    api.runs(
                        filters={"tags": f"{model} Results"},
                    ),
                ),
                key=lambda x: (
                    x.config["dataset"],
                    x.config["split"],
                    x.config["alpha"],
                ),  # type: ignore
            ),
            desc="Getting tables",
        ):
            replacements = {
                "ccsum": "CCSum",
                "cnndm": "CNN/DM",
                "multinews": "Multi-News",
                "scitldr": "SciTLDR",
                "aic": "AIC",
                "abs": "Abs",
                "full": "Full",
            }
            dataset_name = cast(str, run.config["dataset"])
            ref_key, ds = datasets[dataset_name]

            for old_str, new_str in replacements.items():
                dataset_name = dataset_name.replace(old_str, new_str)

            alpha = cast(float, run.config["alpha"])
            row = cast(
                dict[str, float | int | str],
                {
                    "Dataset": dataset_name,
                    "Model": f"{model}{' + BLooP' if alpha else ''}",
                },
            )

            perf_row = row.copy()
            replacements = {
                "rouge1": "ROUGE-1",
                "rouge2": "ROUGE-2",
                "rougeLsum": "ROUGE-Lsum",
                "bart_score_prob": "BARTScore Prob",
                "bart_score": "BARTScore",
            }
            for metric, value in run.summary.items():  # type: ignore
                assert isinstance(metric, str)
                if any(i in metric for i in ["rouge", "bart"]) and not any(
                    i in metric for i in ["precision", "recall"]
                ):
                    for old_str, new_str in replacements.items():
                        metric = metric.replace(old_str, new_str)
                    assert isinstance(value, float)
                    value *= 100 if metric != "BARTScore" else 1
                    perf_row[metric] = round(value, 2)

            df = get_df_from_wandb(api, run, "results")
            predictions: list[str] = []
            references: list[str] = []
            print(f"Dataset: {dataset_name}")

            for orig_ref, (_, df_row) in zip(ds[ref_key], df.iterrows()):  # type: ignore
                pred = cast(str, df_row["pred"])
                if isinstance(orig_ref, list):
                    orig_ref = "\n".join(cast(list[str], orig_ref))
                ref = cast(str, orig_ref)
                predictions.append("\n".join(sent_tokenize(pred)))
                references.append(ref)

            scores_dict = cast(
                dict[str, float],
                rouge_scorer.compute(predictions=predictions, references=references),
            )
            for metric, score in scores_dict.items():
                perf_row[f"hf_{metric}"] = round(100 * score, 2)

            bart_scores = cast(
                list[float],
                bart_scorer.score(srcs=predictions, tgts=references, batch_size=64),
            )
            perf_row["new_bart_score"] = sum(bart_scores) / len(bart_scores)
            bart_scores_prob = [math.exp(score) for score in bart_scores]
            perf_row["new_bart_score_prob"] = sum(bart_scores_prob) / len(
                bart_scores_prob
            )

            perf_data.append(perf_row)
            results_tables[dataset_name][alpha] = get_df_from_wandb(api, run, "results")

            cache_usage_row = row.copy()
            df = get_df_from_wandb(api, run, "cache_usage")
            hit_count = cast(
                int,
                df[df["Event"].str.contains("hit", case=False, na=False)][
                    "gen_steps (num_of)"
                ].item(),
            )
            miss_count = cast(
                int,
                df[df["Event"].str.contains("miss", case=False, na=False)][
                    "gen_steps (num_of)"
                ].item(),
            )
            cache_usage_row["Cache Hit Rate"] = round(
                hit_count / (hit_count + miss_count) * 100, 2
            )
            cache_usage_data.append(cache_usage_row)

            if "BLooP" in row["Model"]:  # type: ignore
                promotion_effect_row = row.copy()
                promotion_effect_row["Model"] = row["Model"].split(" + ")[0]  # type: ignore
                df = get_df_from_wandb(api, run, "promotion_effect")
                pred_changed = cast(
                    float,
                    df[df["Event"].str.contains("changed", case=False, na=False)][
                        "% of gen_steps"
                    ].item(),
                )
                promotion_effect_row["Prediction Change Rate"] = round(pred_changed, 2)
                promotion_effect_data.append(promotion_effect_row)

    for name, table in (
        ("perf_data", perf_data),
        ("cache_usage_data", cache_usage_data),
        ("promotion_effect_data", promotion_effect_data),
    ):
        pd.DataFrame(table).sort_values(["Dataset", "Model"]).to_csv(
            f"{name}.csv", index=False
        )


def main():
    collect_results()


if __name__ == "__main__":
    main()
