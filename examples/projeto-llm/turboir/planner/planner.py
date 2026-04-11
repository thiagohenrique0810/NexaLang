"""Execution Planner — converts TurboIR to concrete execution plan."""

from dataclasses import dataclass, field
from typing import Any

from turboir.ir.nodes import (
    IRProgram, IRModelConfig, QuantType, CacheStrategy,
    AttentionVariant, DecodeMode, SchedulerPolicy, DeviceTarget,
)


@dataclass
class ExecutionStep:
    name: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    order: int = 0


@dataclass
class ExecutionPlan:
    model_name: str = ""
    device: str = "cpu"
    steps: list[ExecutionStep] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        lines = [f"ExecutionPlan: {self.model_name} on {self.device}"]
        lines.append(f"  Steps ({len(self.steps)}):")
        for s in self.steps:
            lines.append(f"    [{s.order}] {s.name}: {s.action} {s.params}")
        return '\n'.join(lines)


class ExecutionPlanner:
    """Transforms TurboIR model configs into ordered execution plans."""

    def plan(self, ir_program: IRProgram) -> list[ExecutionPlan]:
        plans = []
        for model_cfg in ir_program.models:
            plans.append(self._plan_model(model_cfg))
        return plans

    def _plan_model(self, cfg: IRModelConfig) -> ExecutionPlan:
        plan = ExecutionPlan(
            model_name=cfg.name,
            device=self._resolve_device(cfg.target.device),
            config=cfg.to_dict(),
        )

        order = 0

        # Step 1: Load model
        plan.steps.append(ExecutionStep(
            name="load_model",
            action="model_loader.load",
            params={
                'model_name': cfg.name,
                'device': plan.device,
            },
            order=order,
        ))
        order += 1

        # Step 2: Quantize weights (if needed)
        if cfg.weights.quant_type not in (QuantType.NONE, QuantType.FP32):
            plan.steps.append(ExecutionStep(
                name="quantize_weights",
                action="quant.quantize",
                params={
                    'quant_type': cfg.weights.quant_type.value,
                    'strategy': cfg.weights.strategy,
                },
                order=order,
            ))
            order += 1

        # Step 3: Initialize KV cache
        plan.steps.append(ExecutionStep(
            name="init_kv_cache",
            action="kv_cache.init",
            params={
                'strategy': cfg.cache.strategy.value,
                'bits': cfg.cache.bits,
                'max_size': cfg.cache.max_size if cfg.cache.max_size else cfg.scheduler.max_ctx,
            },
            order=order,
        ))
        order += 1

        # Step 4: Configure attention
        plan.steps.append(ExecutionStep(
            name="configure_attention",
            action="attention.configure",
            params={
                'variant': cfg.attention.variant.value,
                'streaming': cfg.attention.streaming,
                'window_size': cfg.attention.window_size,
            },
            order=order,
        ))
        order += 1

        # Step 5: Configure scheduler
        if cfg.scheduler.policy == SchedulerPolicy.CONTINUOUS_BATCHING:
            plan.steps.append(ExecutionStep(
                name="init_scheduler",
                action="scheduler.init_continuous",
                params={
                    'max_batch': cfg.scheduler.max_batch,
                    'max_ctx': cfg.scheduler.max_ctx,
                    'timeout_ms': cfg.scheduler.timeout_ms,
                },
                order=order,
            ))
            order += 1

        # Step 6: Configure decode mode
        if cfg.decode.mode == DecodeMode.SPECULATIVE:
            plan.steps.append(ExecutionStep(
                name="init_speculative",
                action="decode.init_speculative",
                params={
                    'drafter': cfg.decode.drafter,
                    'k': cfg.decode.k,
                },
                order=order,
            ))
            order += 1

        # Step 7: Ready
        plan.steps.append(ExecutionStep(
            name="ready",
            action="engine.ready",
            params={'temperature': cfg.decode.temperature},
            order=order,
        ))

        return plan

    def _resolve_device(self, target: DeviceTarget) -> str:
        if target == DeviceTarget.AUTO:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return target.value
