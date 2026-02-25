from typing import Any, cast

import numpy as np
import pandas as pd
import toml
import wandb
from scipy.stats import false_discovery_control, wilcoxon  # type: ignore
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from utils import get_df_from_wandb


def main():
    with open("dataset_info.toml", "r") as f:
        dataset_info = cast(dict[str, dict[str, str]], toml.load(f))
    datasets = list(dataset_info.keys())
    metrics = ["rouge1", "rouge2", "rougeLsum", "bart_score", "bart_score_prob"]
    models = ["Llama", "Mistral", "Gemma"]
    bloop_scores_diff: dict[str, dict[str, dict[str, pd.Series]]] = {
        dataset: {
            model: {metric: pd.Series() for metric in metrics} for model in models
        }
        for dataset in datasets
    }

    api = wandb.Api()
    for model in models:
        for run in tqdm(
            cast(list[Run], api.runs(filters={"tags": f"{model} Results"})),
            desc="Getting tables",
        ):
            dataset_name = cast(str, run.config["dataset"])
            if dataset_name not in datasets:
                continue

            alpha = cast(float, run.config["alpha"])
            df = get_df_from_wandb(api, run, "results")
            sign = 1 if alpha else -1
            for metric in metrics:
                if bloop_scores_diff[dataset_name][model][metric].empty:
                    bloop_scores_diff[dataset_name][model][metric] = (
                        cast(pd.Series, df[metric]) * sign
                    )
                else:
                    bloop_scores_diff[dataset_name][model][metric] += (
                        cast(pd.Series, df[metric]) * sign
                    )

    p_values = pd.DataFrame(columns=["Dataset", "Model"] + metrics)  # type: ignore
    effect_sizes = pd.DataFrame(columns=["Dataset", "Model"] + metrics)  # type: ignore
    all_p_vals: list[float] = []
    row_idx = 0

    for dataset in datasets:
        for model in models:
            row_p_vals: list[float] = []
            row_effect_sizes: list[float] = []
            for metric in metrics:
                scores = bloop_scores_diff[dataset][model][metric]
                n = len(scores)

                # Perform Wilcoxon signed-rank test
                wilcoxon_result = wilcoxon(scores, alternative="greater")

                p_val = cast(float, wilcoxon_result[1])
                test_statistic = cast(float, wilcoxon_result[0])

                # Calculate rank-biserial correlation (effect size)
                effect_size = 1 - (2 * test_statistic) / (n * (n + 1))

                row_p_vals.append(p_val)
                row_effect_sizes.append(effect_size)
                all_p_vals.append(p_val)

            p_values.loc[row_idx] = [dataset, model, *row_p_vals]
            effect_sizes.loc[row_idx] = [dataset, model, *row_effect_sizes]
            row_idx += 1

    bh_rejected = cast(np.ndarray[np.float64, Any], false_discovery_control(all_p_vals))
    bh_idx = 0
    for row in range(len(p_values)):
        for metric in metrics:
            p_values.loc[row, metric] = bh_rejected[bh_idx]
            bh_idx += 1

    p_values.round(3).to_csv("p_values.csv", index=False)
    effect_sizes.round(3).to_csv("effect_sizes.csv", index=False)


if __name__ == "__main__":
    main()
