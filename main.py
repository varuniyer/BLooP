import gc
from typing import cast

import torch
import wandb
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.outputs import RequestOutput
from wandb.sdk.wandb_run import Run

from analyze import analyze_results
from args import Args, DatasetInfo, create_namespaces
from eval import eval_loop
from logits_process import BLooP
from utils import build_bigram_caches, get_dataset, match_code_files


def run_inference(
    args: Args,
    dataset_info: DatasetInfo,
    wandb_run: Run,
) -> tuple[list[RequestOutput], list[str], list[str], PreTrainedTokenizer]:
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(args.model))
    orig_eos_token_id = tokenizer.eos_token_id  # type: ignore
    if isinstance(orig_eos_token_id, int):
        eos_token_ids = [orig_eos_token_id]
    elif orig_eos_token_id is None:
        eos_token_ids = []
    else:
        raise ValueError(f"Unexpected eos_token_id type: {type(orig_eos_token_id)}")  # type: ignore
    eos_token_ids += [
        token_id
        for token_id in range(tokenizer.vocab_size)
        if "\n" in tokenizer.decode(token_id)
    ]
    all_docs, all_prompts, all_refs = get_dataset(tokenizer, dataset_info, args)

    all_prompt_ids = cast(
        list[list[int]],
        tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in all_prompts],
            add_generation_prompt=True,
        ),
    )
    del all_prompts
    all_prompt_lens = [len(ids) for ids in all_prompt_ids]
    max_prompt_len = max(all_prompt_lens)
    print(f"Max prompt len: {max_prompt_len}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=args.model_seed,
        max_model_len=max_prompt_len * 3 // 2,
        gpu_memory_utilization=0.95,
    )

    logits_processors = [
        BLooP(args, bigram_cache, eos_token_ids)
        for bigram_cache in build_bigram_caches(all_docs, tokenizer)
    ]

    sampling_params = [
        SamplingParams(
            best_of=args.beam_width,
            use_beam_search=args.beam_width > 1,
            temperature=0,
            top_p=1,
            max_tokens=doc_len // 2,
            logits_processors=[logits_processor],
            seed=args.model_seed,
            stop=[".\n"],
        )
        for doc_len, logits_processor in zip(all_prompt_lens, logits_processors)
    ]

    all_outputs = llm.generate(
        sampling_params=sampling_params, prompt_token_ids=all_prompt_ids
    )
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    analyze_results(tokenizer, logits_processors, all_outputs, wandb_run, args.profile)

    return all_outputs, all_docs, all_refs, tokenizer


if __name__ == "__main__":
    args, dataset_info = create_namespaces()

    assert 0 <= args.subsample <= 1, "Subsample probability must be in [0, 1]"

    wandb_run = wandb.init(config=vars(args))
    wandb_run.log_code(".", include_fn=match_code_files)

    eval_args = run_inference(args, dataset_info, wandb_run)
    df, avg_scores = eval_loop(*eval_args)
    table = wandb.Table(dataframe=df)

    wandb_run.summary.update(avg_scores)
    wandb_run.log({"results": table})
    wandb_run.finish()
