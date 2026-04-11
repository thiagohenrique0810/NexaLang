"""Prefill — processes the full prompt in one forward pass."""

import logging
from typing import Any

logger = logging.getLogger("turboruntime.prefill")


class Prefill:
    def __init__(self, model: Any, device: str = "cpu"):
        self.model = model
        self.device = device

    def run(self, input_ids: Any, attention_mask: Any = None) -> tuple[Any, Any]:
        """Run prefill on input_ids.

        Returns:
            (logits, past_key_values): The model output logits and KV cache tensors.
        """
        import torch

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                use_cache=True,
            )

        return outputs.logits, outputs.past_key_values
