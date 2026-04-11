"""Decoder — autoregressive token generation with sampling."""

import logging
from typing import Any

logger = logging.getLogger("turboruntime.decode")


class Decoder:
    def __init__(self, model: Any, device: str = "cpu"):
        self.model = model
        self.device = device

    def step(self, input_id: Any, past_key_values: Any,
             attention_mask: Any = None) -> tuple[Any, Any]:
        """Run one decode step.

        Args:
            input_id: [batch, 1] tensor of the last generated token
            past_key_values: KV cache from previous steps
            attention_mask: Optional attention mask

        Returns:
            (logits, past_key_values)
        """
        import torch

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_id.to(self.device),
                past_key_values=past_key_values,
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                use_cache=True,
            )

        return outputs.logits, outputs.past_key_values

    @staticmethod
    def sample(logits: Any, temperature: float = 1.0,
               top_p: float = 1.0, top_k: int = 0) -> Any:
        """Sample a token from logits with temperature, top-p, and top-k."""
        import torch

        logits = logits[:, -1, :].float()

        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class SpeculativeDecoder:
    """Speculative decoding — use a small draft model to predict k tokens,
    then verify with the main model in one pass."""

    def __init__(self, main_model: Any, draft_model: Any = None,
                 k: int = 4, device: str = "cpu"):
        self.main = main_model
        self.draft = draft_model
        self.k = k
        self.device = device
        self._enabled = draft_model is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def decode_step(self, input_ids: Any, past_main: Any, past_draft: Any,
                    temperature: float = 1.0) -> tuple[Any, Any, Any, int]:
        """Run speculative decode: draft k tokens, verify with main model.

        Returns:
            (accepted_tokens, new_past_main, new_past_draft, num_accepted)
        """
        import torch

        if not self._enabled:
            # Fallback to standard decode
            decoder = Decoder(self.main, self.device)
            logits, past_main = decoder.step(input_ids[:, -1:], past_main)
            token = Decoder.sample(logits, temperature=temperature)
            return token, past_main, past_draft, 1

        # Draft k tokens
        draft_tokens = []
        draft_input = input_ids[:, -1:]
        current_past = past_draft

        for _ in range(self.k):
            with torch.no_grad():
                out = self.draft(
                    input_ids=draft_input.to(self.device),
                    past_key_values=current_past,
                    use_cache=True,
                )
            token = Decoder.sample(out.logits, temperature=temperature)
            draft_tokens.append(token)
            draft_input = token
            current_past = out.past_key_values

        # Verify with main model (single forward pass for all k tokens)
        draft_seq = torch.cat(draft_tokens, dim=1)
        verify_input = torch.cat([input_ids[:, -1:], draft_seq], dim=1)

        with torch.no_grad():
            main_out = self.main(
                input_ids=verify_input.to(self.device),
                past_key_values=past_main,
                use_cache=True,
            )

        # Accept tokens where main model agrees
        accepted = []
        for i in range(self.k):
            main_token = Decoder.sample(main_out.logits[:, i:i+1, :], temperature=temperature)
            if i < len(draft_tokens) and main_token.item() == draft_tokens[i].item():
                accepted.append(draft_tokens[i])
            else:
                accepted.append(main_token)
                break

        num_accepted = len(accepted)
        result = torch.cat(accepted, dim=1)

        return result, main_out.past_key_values, current_past, num_accepted
