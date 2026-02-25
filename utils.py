import json
import random
import re
import string
from collections import Counter, defaultdict
from tempfile import TemporaryDirectory
from typing import Generator, Mapping, cast

import pandas as pd
import wandb
from datasets import (
    arrow_dataset,
    load_dataset,  # type: ignore
)
from nltk.tokenize import sent_tokenize  # type: ignore
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from wandb.apis.public.api import Api
from wandb.sdk.wandb_run import Run

from args import Args, DatasetInfo


def match_code_files(path: str) -> bool:
    return bool(re.match(r".*\.(py|yml|sh)$", path))


def strip_punct_space(text: str) -> str:
    return (
        text.translate(str.maketrans("", "", string.punctuation))
        .replace("\n", "")
        .strip()
    )


def log_data_with_percent(
    data: Mapping[str, int | float], table_name: str, columns: list[str], wandb_run: Run
) -> None:
    columns.append(f"% of {columns[1].split()[0]}")
    total = sum(data.values())
    data_as_list = [[k, v, v / total * 100] for k, v in data.items()]
    data_table = wandb.Table(data=data_as_list, columns=columns)
    wandb_run.log({table_name: data_table})


def get_df_from_wandb(api: Api, run: Run, table_name: str) -> pd.DataFrame:
    with TemporaryDirectory() as tmpdir:
        fname = f"{tmpdir}/{table_name}.table.json"
        api.artifact(f"run-{run.id}-{table_name}:latest").download(root=tmpdir)
        with open(fname, "r", encoding="utf-8") as f:
            df_data = json.load(f)

    return pd.DataFrame(df_data["data"], columns=df_data["columns"])


def truncate_docs(
    tokenizer: PreTrainedTokenizer,
    template: str,
    max_input_len: int,
    docs: list[str],
) -> list[str]:
    doc_ids = cast(
        list[list[int]], tokenizer(docs, add_special_tokens=False)["input_ids"]
    )
    prompts = [[{"role": "user", "content": template.format(article=d)}] for d in docs]
    prompt_ids = cast(list[list[int]], tokenizer.apply_chat_template(prompts))
    trunc_lens = [max(0, len(ids) - max_input_len) for ids in prompt_ids]

    return [
        tokenizer.decode(ids[:-trunc_len]) if trunc_len else doc
        for doc, ids, trunc_len in zip(docs, doc_ids, trunc_lens)
    ]


def get_prompts(
    all_docs: list[str], tokenizer: PreTrainedTokenizer, args: Args
) -> list[str]:
    prompt_template = "Write a paragraph summarizing the following article (without including any text or explanation outside the summary):\n\n{article}"

    if args.max_input_len != -1:
        prompt_docs = truncate_docs(
            tokenizer, prompt_template, args.max_input_len, all_docs
        )
    else:
        prompt_docs = all_docs

    return [prompt_template.format(article=doc) for doc in prompt_docs]


def get_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_info: DatasetInfo,
    args: Args,
) -> tuple[list[str], list[str], list[str]]:
    dataset_name = dataset_info.name
    subset = dataset_info.subset or None
    ds = load_dataset(
        dataset_name, name=subset, split=args.split, trust_remote_code=True
    )

    ds = cast(arrow_dataset.Dataset, ds)
    ds_docs = cast(list[str | list[str]], ds[dataset_info.doc_key])
    ds_refs = cast(list[str | list[str]], ds[dataset_info.ref_key])

    all_docs: list[str] = []
    all_refs: list[str] = []

    num_empty_docs = 0
    num_empty_refs = 0
    for doc, ref in zip(ds_docs, ds_refs):
        num_empty_docs += not doc
        num_empty_refs += not ref
        if doc and ref:
            # newlines between ref sentences are needed for rougeLsum
            if "scitldr" in dataset_name:
                all_docs.append(" ".join(doc))
                all_refs.append("\n".join(ref))
            else:
                assert isinstance(doc, str) and isinstance(ref, str), (
                    "docs and refs must be strings"
                )
                all_docs.append(doc)
                all_refs.append(ref)

    print(f"Number of empty docs: {num_empty_docs}/{len(ds)}")
    print(f"Number of empty refs: {num_empty_refs}/{len(ds)}")

    all_docs = [doc.strip() for doc in all_docs]
    all_prompts = get_prompts(all_docs, tokenizer, args)
    all_refs = [ref.strip() for ref in all_refs]

    if args.subsample < 1:
        random.seed(args.data_seed)
        indices = random.sample(
            range(len(all_docs)), int(len(all_docs) * args.subsample)
        )
        indices.sort()
        all_docs = [all_docs[i] for i in indices]
        all_prompts = [all_prompts[i] for i in indices]
        all_refs = [all_refs[i] for i in indices]
        random.seed(args.model_seed)

    return all_docs, all_prompts, all_refs


BigramCache = dict[int, list[tuple[int, int]]]


def build_bigram_caches(
    all_docs: list[str], tokenizer: PreTrainedTokenizer
) -> Generator[BigramCache, None, None]:
    for doc in tqdm(all_docs, desc="Building data structures"):
        doc_sents = [sent.strip() for sent in sent_tokenize(doc)]
        doc_tokens = cast(
            list[list[int]], tokenizer(doc_sents, add_special_tokens=False).input_ids
        )

        bigram_cache: dict[int, Counter[int]] = defaultdict(Counter)
        for sent_tokens in doc_tokens:
            for token_id, next_token_id in zip(sent_tokens, sent_tokens[1:]):
                bigram_cache[token_id].update([next_token_id])
        yield {k: sorted(v.items()) for k, v in bigram_cache.items()}
