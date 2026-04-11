"""Model Loader — loads HuggingFace models with configurable precision."""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("turboruntime.model_loader")


@dataclass
class LoadedModel:
    name: str = ""
    model: Any = None
    tokenizer: Any = None
    config: Any = None
    device: str = "cpu"
    dtype: Any = None
    num_layers: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0


class ModelLoader:
    def load(self, model_name: str, device: str = "cpu", dtype: str = "fp16") -> LoadedModel:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        logger.info(f"Loading model: {model_name} on {device} with {dtype}")

        torch_dtype = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }.get(dtype, torch.float16)

        # Adjust for MPS — bf16 not supported
        if device == "mps" and torch_dtype == torch.bfloat16:
            torch_dtype = torch.float16

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device if device != "mps" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if device == "mps":
            model = model.to(device)

        model.eval()

        # Extract architecture info
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', 768))
        num_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 12))
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
        head_dim = hidden_size // num_heads
        vocab_size = getattr(config, 'vocab_size', 50257)

        loaded = LoadedModel(
            name=model_name,
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            dtype=torch_dtype,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
        )

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {param_count/1e6:.1f}M params, {num_layers} layers, "
                     f"hidden={hidden_size}, heads={num_heads}")
        return loaded
