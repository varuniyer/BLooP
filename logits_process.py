from collections import defaultdict
from time import perf_counter
from typing import cast

import torch

from args import Args
from utils import BigramCache


class BLooP:
    def __init__(
        self,
        args: Args,
        bigram_cache: BigramCache,
        eos_token_ids: list[int],
    ):
        self.profile = args.profile
        self.use_fw = args.use_fw
        self.alpha = args.alpha
        self.bigram_cache = bigram_cache

        self.eos_token_ids = set(eos_token_ids)
        self.op_time: dict[str, float] = defaultdict(float)
        self.cache_hits: set[tuple[int, ...]] = set()
        self.changed_preds: set[tuple[int, ...]] = set()

    def _profile(self, name: str):
        if not self.profile:
            return
        torch.cuda.synchronize()
        self.op_time[name] += (cur_time := perf_counter()) - self.start
        self.start = cur_time

    def __call__(
        self, input_ids: tuple[int, ...], scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.start = perf_counter()
        if not len(input_ids):
            return scores
        self._profile("check_empty")

        initial_pred = cast(int, scores.argmax().item())
        self._profile("get_initial_pred")

        if initial_pred in self.eos_token_ids:
            return scores
        self._profile("check_eos")

        if (prev_token := input_ids[-1]) not in self.bigram_cache:
            return scores

        self._profile("check_cache_presence")
        self.cache_hits.add(input_ids)

        for token_id, freq in self.bigram_cache[prev_token]:
            scores[token_id] += self.alpha * (freq if self.use_fw else 1)
        self._profile("apply_promotion")

        final_pred = cast(int, scores.argmax().item())
        if final_pred != initial_pred:
            self.changed_preds.add(input_ids)

        self._profile("check_pred_change")

        return scores
