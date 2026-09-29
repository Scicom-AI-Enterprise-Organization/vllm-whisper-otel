# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constrain the first output tokens of a request to given token sets.

A request opts in with ``SamplingParams.extra_args["forced_prefix"]``: entry
``i`` lists the token ids allowed at output position ``i``. One id forces that
token. Several ids restrict the choice to them and always take the one with
the highest logit, whatever the temperature, which is how Whisper picks its
language token.

Enable it with::

    --logits-processors \\
        vllm.v1.sample.logits_processor.forced_prefix:ForcedPrefixLogitsProcessor

Model Runner V2 does not run logits processors, so this flag also makes the
engine use the V1 model runner.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig

FORCED_PREFIX_KEY = "forced_prefix"


def get_forced_prefix(params: SamplingParams) -> list[list[int]] | None:
    return (params.extra_args or {}).get(FORCED_PREFIX_KEY)


class ForcedPrefixLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        prefix = get_forced_prefix(sampling_params)
        if prefix is None:
            return
        if not (
            isinstance(prefix, list)
            and prefix
            and all(
                isinstance(step, list)
                and step
                and all(isinstance(tok, int) for tok in step)
                for step in prefix
            )
        ):
            raise VLLMValidationError(
                f"extra_args.{FORCED_PREFIX_KEY} must be a non-empty list of "
                "non-empty lists of token ids",
                parameter="extra_args",
            )

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        self.device = device
        # batch index -> (allowed ids per output position, running output ids)
        self.prefixes: dict[int, tuple[list[torch.Tensor], Sequence[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def _add_request(
        self,
        params: SamplingParams,
        _: list[int] | None,
        output_tok_ids: list[int],
    ) -> tuple[list[torch.Tensor], Sequence[int]] | None:
        prefix = get_forced_prefix(params)
        if not prefix or len(output_tok_ids) >= len(prefix):
            return None
        steps = [
            torch.tensor(step, dtype=torch.long, device=self.device) for step in prefix
        ]
        return steps, output_tok_ids

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self.prefixes, batch_update, self._add_request)
        done = [
            index
            for index, (steps, output_tok_ids) in self.prefixes.items()
            if len(output_tok_ids) >= len(steps)
        ]
        for index in done:
            del self.prefixes[index]

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, (steps, output_tok_ids) in self.prefixes.items():
            allowed = steps[len(output_tok_ids)]
            row = logits[index]
            chosen = allowed[row[allowed].argmax()] if len(allowed) > 1 else allowed
            row.fill_(float("-inf"))
            row.index_fill_(0, chosen.view(1), 0.0)
        return logits
