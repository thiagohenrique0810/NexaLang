"""TurboRuntime Engine — main inference engine that orchestrates all components."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator

from turboruntime.model_loader.loader import ModelLoader, LoadedModel
from turboruntime.quant.quantizer import Quantizer
from turboruntime.kv_cache.cache import KVCache
from turboruntime.scheduler.scheduler import Scheduler, InferenceRequest, RequestState
from turboruntime.prefill.prefill import Prefill
from turboruntime.decode.decoder import Decoder, SpeculativeDecoder
from turboir.ir.nodes import QuantType, CacheStrategy, DecodeMode
from turboir.planner.planner import ExecutionPlan

logger = logging.getLogger("turboruntime.engine")


@dataclass
class GenerationResult:
    text: str = ""
    tokens_generated: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_per_sec: float = 0.0
    prompt_tokens: int = 0


@dataclass
class EngineConfig:
    model_name: str = "gpt2"
    device: str = "cpu"
    quant_type: str = "none"
    cache_strategy: str = "standard"
    cache_bits: int = 16
    max_ctx: int = 2048
    max_batch: int = 8
    decode_mode: str = "autoregressive"
    drafter_model: str = ""
    speculative_k: int = 4
    temperature: float = 1.0

    @classmethod
    def from_plan(cls, plan: ExecutionPlan) -> 'EngineConfig':
        cfg = plan.config
        return cls(
            model_name=cfg['name'],
            device=plan.device,
            quant_type=cfg.get('weights', {}).get('quant_type', 'none'),
            cache_strategy=cfg.get('cache', {}).get('strategy', 'standard'),
            cache_bits=cfg.get('cache', {}).get('bits', 16),
            max_ctx=cfg.get('scheduler', {}).get('max_ctx', 2048),
            max_batch=cfg.get('scheduler', {}).get('max_batch', 8),
            decode_mode=cfg.get('decode', {}).get('mode', 'autoregressive'),
            drafter_model=cfg.get('decode', {}).get('drafter', ''),
            speculative_k=cfg.get('decode', {}).get('k', 4),
            temperature=cfg.get('decode', {}).get('temperature', 1.0),
        )


class InferenceEngine:
    def __init__(self):
        self.model: LoadedModel | None = None
        self.scheduler: Scheduler | None = None
        self.kv_cache: KVCache | None = None
        self.decoder: Decoder | None = None
        self.prefill: Prefill | None = None
        self.spec_decoder: SpeculativeDecoder | None = None
        self.config: EngineConfig | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def init_from_plan(self, plan: ExecutionPlan):
        config = EngineConfig.from_plan(plan)
        self.init(config)

    def init(self, config: EngineConfig):
        self.config = config
        logger.info(f"Initializing engine: model={config.model_name}, device={config.device}")

        # 1. Load model
        loader = ModelLoader()
        dtype = "fp16"
        if config.quant_type in ("int4", "int8"):
            dtype = "fp32"  # Load full precision before quantizing
        self.model = loader.load(config.model_name, device=config.device, dtype=dtype)

        # 2. Quantize if needed
        if config.quant_type in ("int4", "int8"):
            quantizer = Quantizer()
            self.model.model, stats = quantizer.quantize(
                self.model.model, quant_type=config.quant_type, device=config.device
            )
            logger.info(f"Quantization stats: {stats}")

        # 3. Init KV cache (managed by HF model internally for now)
        # External cache used for metrics tracking
        self.kv_cache = KVCache(
            num_layers=self.model.num_layers,
            num_heads=self.model.num_heads,
            head_dim=self.model.head_dim,
            max_seq_len=config.max_ctx,
            device=config.device,
            dtype=self.model.dtype,
            compression=config.cache_strategy,
            bits=config.cache_bits,
        )

        # 4. Init scheduler
        self.scheduler = Scheduler(
            max_batch=config.max_batch,
            max_ctx=config.max_ctx,
        )

        # 5. Init prefill and decoder
        self.prefill = Prefill(self.model.model, device=config.device)
        self.decoder = Decoder(self.model.model, device=config.device)

        self._ready = True
        logger.info("Engine ready")

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = None, top_p: float = 1.0,
                 top_k: int = 0) -> GenerationResult:
        """Generate text from a prompt (synchronous, single request)."""
        import torch

        if not self._ready:
            raise RuntimeError("Engine not initialized. Call init() first.")

        if temperature is None:
            temperature = self.config.temperature

        start_time = time.time()
        tokenizer = self.model.tokenizer

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs.input_ids.to(self.config.device)
        attention_mask = inputs.attention_mask.to(self.config.device)
        prompt_len = input_ids.shape[1]

        # Prefill
        logits, past_key_values = self.prefill.run(input_ids, attention_mask)
        first_token = Decoder.sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        ttft = (time.time() - start_time) * 1000

        generated = [first_token]
        current_past = past_key_values
        current_mask = torch.cat([
            attention_mask,
            torch.ones(1, 1, dtype=torch.long, device=self.config.device)
        ], dim=1)

        # Decode loop
        for _ in range(max_tokens - 1):
            next_logits, current_past = self.decoder.step(
                generated[-1], current_past, current_mask
            )
            next_token = Decoder.sample(
                next_logits, temperature=temperature, top_p=top_p, top_k=top_k
            )

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated.append(next_token)
            current_mask = torch.cat([
                current_mask,
                torch.ones(1, 1, dtype=torch.long, device=self.config.device)
            ], dim=1)

        total_time = (time.time() - start_time) * 1000
        num_generated = len(generated)

        all_tokens = torch.cat(generated, dim=1)
        text = tokenizer.decode(all_tokens[0], skip_special_tokens=True)

        decode_time = total_time - ttft
        tokens_per_sec = num_generated / (decode_time / 1000) if decode_time > 0 else 0

        result = GenerationResult(
            text=text,
            tokens_generated=num_generated,
            ttft_ms=ttft,
            total_ms=total_time,
            tokens_per_sec=tokens_per_sec,
            prompt_tokens=prompt_len,
        )

        logger.info(f"Generated {num_generated} tokens in {total_time:.0f}ms "
                     f"(TTFT={ttft:.0f}ms, {tokens_per_sec:.1f} tok/s)")
        return result

    def generate_stream(self, prompt: str, max_tokens: int = 128,
                        temperature: float = None, top_p: float = 1.0,
                        top_k: int = 0) -> Generator[str, None, None]:
        """Generate text token-by-token (streaming)."""
        import torch

        if not self._ready:
            raise RuntimeError("Engine not initialized.")

        if temperature is None:
            temperature = self.config.temperature

        tokenizer = self.model.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs.input_ids.to(self.config.device)
        attention_mask = inputs.attention_mask.to(self.config.device)

        logits, past_key_values = self.prefill.run(input_ids, attention_mask)
        token = Decoder.sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        yield tokenizer.decode(token[0], skip_special_tokens=True)

        current_past = past_key_values
        current_mask = torch.cat([
            attention_mask,
            torch.ones(1, 1, dtype=torch.long, device=self.config.device)
        ], dim=1)

        for _ in range(max_tokens - 1):
            logits, current_past = self.decoder.step(token, current_past, current_mask)
            token = Decoder.sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)

            if token.item() == tokenizer.eos_token_id:
                break

            yield tokenizer.decode(token[0], skip_special_tokens=True)
            current_mask = torch.cat([
                current_mask,
                torch.ones(1, 1, dtype=torch.long, device=self.config.device)
            ], dim=1)

    def get_metrics(self) -> dict:
        metrics = {
            'ready': self._ready,
            'model': self.config.model_name if self.config else None,
            'device': self.config.device if self.config else None,
        }
        if self.kv_cache:
            metrics['kv_cache'] = {
                'entries': self.kv_cache.stats.total_entries,
                'memory_mb': self.kv_cache.stats.memory_used_mb,
                'compression_ratio': self.kv_cache.stats.compression_ratio,
            }
        if self.scheduler:
            metrics['scheduler'] = self.scheduler.metrics
        return metrics

    def shutdown(self):
        logger.info("Shutting down engine")
        if self.kv_cache:
            self.kv_cache.clear()
        self.model = None
        self._ready = False
