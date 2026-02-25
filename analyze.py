from collections import Counter, defaultdict
from typing import cast

import matplotlib.pyplot as plt
import spacy
import wandb
from matplotlib.axes import Axes
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from vllm.outputs import RequestOutput
from wandb.sdk.wandb_run import Run

from logits_process import BLooP
from utils import log_data_with_percent, strip_punct_space


def analyze_results(
    tokenizer: PreTrainedTokenizer,
    logits_processors: list[BLooP],
    all_outputs: list[RequestOutput],
    wandb_run: Run,
    profile: bool,
):
    all_cache_usage: Counter[str] = Counter()
    all_promotion_effect: Counter[str] = Counter()
    all_op_times: dict[str, float] = defaultdict(float)

    changed_preds_relative_position: list[int] = [0] * 101
    preds_relative_position: list[int] = [0] * 101
    all_changed_pos_tags: Counter[str] = Counter()
    all_pos_tags: Counter[str] = Counter()
    alignment_failures, dataset_size = 0, len(all_outputs)

    nlp = spacy.load("en_core_web_trf")

    for output, logits_processor in tqdm(
        zip(all_outputs, logits_processors),
        total=dataset_size,
        desc="Analyzing results",
    ):
        for k, v in logits_processor.op_time.items():
            all_op_times[k] += v

        out_tokens: tuple[int, ...] = tuple()
        changed_pos_tags: Counter[str] = Counter()
        output_str = output.outputs[0].text
        pos_tags: list[str] = []
        spacy_tokens: list[str] = []
        changed_spacy_tokens: list[bool] = []

        for token in nlp(output_str):
            word = strip_punct_space(token.text)
            if word != "":
                pos_tags.append(token.pos_)
                spacy_tokens.append(word)
        all_pos_tags.update(pos_tags)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)
        summary_length = len(output_ids)

        for token_idx in range(summary_length):
            preds_relative_position[round(100 * token_idx / summary_length)] += 1

        decoded_spacy_tokens: list[str] = []
        did_token_change = False
        spacy_token_idx = 0
        out_tokens = tuple()
        alignment_complete = False
        token_idx_to_spacy_idx: dict[int, int] = {}
        for token_idx, token_id in enumerate(output_ids):
            if out_tokens in logits_processor.cache_hits:
                all_cache_usage.update(["cache_hit"])
            else:
                all_cache_usage.update(["cache_miss"])

            if out_tokens in logits_processor.changed_preds:
                all_promotion_effect.update(["promotion_changed_prediction"])
            else:
                all_promotion_effect.update(["promotion_didnt_change_prediction"])
            out_tokens += (token_id,)

            token_str = strip_punct_space(tokenizer.decode(token_id))
            did_token_change = (
                did_token_change or out_tokens in logits_processor.changed_preds
            )

            if not alignment_complete:
                for spacy_subtoken in nlp.tokenizer(token_str):
                    decoded_spacy_tokens.append(spacy_subtoken.text)
                    spacy_token = strip_punct_space("".join(decoded_spacy_tokens))

                    if spacy_token_idx >= len(spacy_tokens):
                        alignment_complete = True
                        break

                    target_token = spacy_tokens[spacy_token_idx]

                    if spacy_token.startswith(target_token):
                        token_idx_to_spacy_idx[token_idx] = spacy_token_idx
                        spacy_token_idx += 1
                        changed_spacy_tokens.append(did_token_change)
                        did_token_change = False

                        remaining = spacy_token[len(target_token) :]
                        if remaining:
                            decoded_spacy_tokens = [remaining]
                        else:
                            decoded_spacy_tokens = []
                    elif len(spacy_token.split()) > len(target_token.split()):
                        alignment_failures += 1
                        break

        if len(changed_spacy_tokens) != len(spacy_tokens):
            alignment_failures += 1
            continue

        for token_idx, spacy_idx in token_idx_to_spacy_idx.items():
            if (
                spacy_idx < len(changed_spacy_tokens)
                and changed_spacy_tokens[spacy_idx]
            ):
                changed_preds_relative_position[
                    round(100 * token_idx / summary_length)
                ] += 1
                if spacy_idx < len(pos_tags):
                    changed_pos_tags.update([pos_tags[spacy_idx]])

        all_changed_pos_tags.update(changed_pos_tags)

    log_data_with_percent(
        all_cache_usage,
        "cache_usage",
        ["Event", "gen_steps (num_of)"],
        wandb_run,
    )
    log_data_with_percent(
        all_promotion_effect,
        "promotion_effect",
        ["Event", "gen_steps (num_of)"],
        wandb_run,
    )
    if profile:
        log_data_with_percent(
            all_op_times,
            "logit_proc_operation_times",
            ["Operation", "Time (s)"],
            wandb_run,
        )

    if any(changed_preds_relative_position):
        percent_changed = [
            round(
                changed_preds_relative_position[position]
                / preds_relative_position[position]
                * 100
            )
            for position in range(101)
        ]

        plt.bar(
            list(range(101)),
            percent_changed,
            width=1,
            align="center",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
        )

        xticks = list(range(0, 101, 10))
        xtick_labels = [f"{x}%" for x in xticks]

        plt.xticks(xticks, xtick_labels, rotation=0)
        plt.grid(True, linestyle="-", alpha=0.3, color="gray", linewidth=0.5)
        plt.xlabel("Position through summary (%)")
        plt.ylabel("Tokens affected by BLooP (%)")
        plt.title("Percentage of tokens affected by BLooP by position through summary")
        plt.tight_layout()
        wandb_run.log({"relative_position_histogram": wandb.Image(plt.gcf())})
        plt.close()

        # Plot POS tag distribution
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = cast(tuple[Axes, Axes], axes)

        top_pos_tags, top_pos_tags_counts = zip(*sorted(all_pos_tags.most_common(9)))
        other_counts = all_pos_tags.total() - sum(top_pos_tags_counts)
        ax1.pie(
            top_pos_tags_counts + (other_counts,),
            labels=top_pos_tags + ("Other",),
            autopct="%.0f%%",
            startangle=90,
        )
        ax1.set_title("All Part-of-Speech tags")

        top_pos_tags_changed_counts = tuple(
            all_changed_pos_tags[tag] for tag in top_pos_tags
        )
        other_counts = all_changed_pos_tags.total() - sum(top_pos_tags_changed_counts)

        ax2.pie(
            top_pos_tags_changed_counts + (other_counts,),
            labels=top_pos_tags + ("Other",),
            autopct="%.0f%%",
            startangle=90,
        )
        ax2.set_title("Part-of-Speech tags of tokens affected by BLooP")

        plt.subplots_adjust(wspace=0.1)
        plt.tight_layout()
        wandb_run.log({"pos_tag_pie_charts": wandb.Image(plt.gcf())})
        plt.close()

        percent_top_pos_tags_changed = [
            count / total * 100 if total else 0
            for count, total in zip(top_pos_tags_changed_counts, top_pos_tags_counts)
        ]
        plt.bar(
            top_pos_tags,
            percent_top_pos_tags_changed,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
        )
        plt.grid(True, axis="y", linestyle="-", alpha=0.3, color="gray", linewidth=0.5)
        plt.xlabel("Part-of-Speech tag")
        plt.ylabel("% of tokens with this POS tag affected by BLooP")
        plt.title("% of tokens with each POS tag affected by BLooP")
        wandb_run.log({"pos_tag_bar_chart": wandb.Image(plt.gcf())})
        plt.close()

    wandb_run.summary["alignment_failure_rate_percent"] = (
        alignment_failures / dataset_size * 100
    )
