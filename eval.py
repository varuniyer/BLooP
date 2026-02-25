import math
from collections import defaultdict
from typing import cast

import evaluate
import pandas as pd
import torch
from nltk import sent_tokenize  # type: ignore
from transformers.tokenization_utils import PreTrainedTokenizer
from vllm.outputs import RequestOutput

from bart_score import BARTScorer


def eval_loop(
    all_outputs: list[RequestOutput],
    all_docs: list[str],
    all_refs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model_io: dict[str, list[str]] = defaultdict(list)
    for doc, ref, req_output in zip(all_docs, all_refs, all_outputs):
        output = req_output.outputs[0]
        model_io["pred"].append("\n".join(sent_tokenize(output.text)))
        prompt = tokenizer.decode(req_output.prompt_token_ids)
        model_io["prompt"].append(prompt)
        model_io["doc"].append(doc)
        model_io["ref"].append(ref)

    all_scores: dict[str, list[float]] = defaultdict(list)

    avg_scores: dict[str, float] = defaultdict(float)
    rouge = evaluate.load("rouge")
    rouge_scores_agg: dict[str, float] = cast(
        dict[str, float],
        rouge.compute(
            predictions=model_io["pred"],
            references=model_io["ref"],
        ),
    )
    for metric, score in rouge_scores_agg.items():
        avg_scores[metric] = score

    rouge_scores_all: dict[str, list[float]] = cast(
        dict[str, list[float]],
        rouge.compute(
            predictions=model_io["pred"],
            references=model_io["ref"],
            use_aggregator=False,
        ),
    )
    for metric, scores in rouge_scores_all.items():
        all_scores[metric] = scores

    bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
    scores = cast(
        list[float],
        bart_scorer.score(srcs=model_io["pred"], tgts=model_io["ref"], batch_size=64),
    )
    probs = [math.exp(score) for score in scores]
    bart_scores = {"bart_score": scores, "bart_score_prob": probs}

    for metric, scores in bart_scores.items():
        all_scores[metric] = scores
        avg_scores[metric] = sum(scores) / len(scores)

    del bart_scorer
    torch.cuda.empty_cache()

    df_out = pd.DataFrame(model_io)
    for metric, scores in all_scores.items():
        df_out[metric] = scores
    return df_out, avg_scores
