"""Weight Quantizer — INT4/INT8 quantization for model weights."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("turboruntime.quant")


@dataclass
class QuantStats:
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 1.0
    num_quantized_layers: int = 0


class Quantizer:
    def __init__(self):
        self.stats = QuantStats()

    def quantize(self, model: Any, quant_type: str = "int8", device: str = "cpu") -> tuple[Any, QuantStats]:
        import torch
        import torch.nn as nn

        logger.info(f"Quantizing model with {quant_type}")

        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        self.stats.original_size_mb = original_size

        if quant_type == "int8":
            model = self._quantize_int8(model, device)
        elif quant_type == "int4":
            model = self._quantize_int4(model, device)
        else:
            logger.warning(f"Unknown quant type {quant_type}, skipping")
            return model, self.stats

        quantized_size = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)
        self.stats.quantized_size_mb = quantized_size
        self.stats.compression_ratio = original_size / max(quantized_size, 0.01)

        logger.info(f"Quantization complete: {original_size:.1f}MB -> {quantized_size:.1f}MB "
                     f"({self.stats.compression_ratio:.1f}x compression)")
        return model, self.stats

    def _quantize_int8(self, model: Any, device: str) -> Any:
        import torch
        import torch.nn as nn

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data.float()
                scale = weight.abs().max() / 127.0
                if scale == 0:
                    continue
                quantized = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
                # Store quantized weight and scale for dequantization
                module.weight.data = (quantized.float() * scale).to(module.weight.dtype)
                module._quant_scale = scale
                module._quant_type = "int8"
                count += 1
        self.stats.num_quantized_layers = count
        return model

    def _quantize_int4(self, model: Any, device: str) -> Any:
        import torch
        import torch.nn as nn

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data.float()
                scale = weight.abs().max() / 7.0
                if scale == 0:
                    continue
                quantized = torch.clamp(torch.round(weight / scale), -8, 7)
                module.weight.data = (quantized * scale).to(module.weight.dtype)
                module._quant_scale = scale
                module._quant_type = "int4"
                count += 1
        self.stats.num_quantized_layers = count
        return model

    def dequantize_tensor(self, tensor: Any, scale: float, quant_type: str = "int8") -> Any:
        import torch
        return tensor.float() * scale
