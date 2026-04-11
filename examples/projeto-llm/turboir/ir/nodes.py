"""TurboIR — Intermediate Representation nodes and AST-to-IR lowering."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from turbolang.ast.nodes import Program, ModelBlock


class QuantType(Enum):
    NONE = "none"
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class CacheStrategy(Enum):
    STANDARD = "standard"
    TURBOQUANT = "turboquant"
    PAGED = "paged"
    NONE = "none"


class AttentionVariant(Enum):
    STANDARD = "standard"
    FLASH = "flash"
    LINEAR = "linear"
    SLIDING_WINDOW = "sliding_window"


class DecodeMode(Enum):
    AUTOREGRESSIVE = "autoregressive"
    SPECULATIVE = "speculative"
    PARALLEL = "parallel"


class SchedulerPolicy(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    CONTINUOUS_BATCHING = "continuous_batching"


class DeviceTarget(Enum):
    GPU = "gpu"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class IRWeightsConfig:
    quant_type: QuantType = QuantType.FP16
    strategy: str = "full"


@dataclass
class IRCacheConfig:
    strategy: CacheStrategy = CacheStrategy.STANDARD
    bits: int = 16
    max_size: int = 0


@dataclass
class IRAttentionConfig:
    variant: AttentionVariant = AttentionVariant.STANDARD
    streaming: bool = False
    window_size: int = 0


@dataclass
class IRDecodeConfig:
    mode: DecodeMode = DecodeMode.AUTOREGRESSIVE
    drafter: str = ""
    k: int = 5
    temperature: float = 1.0


@dataclass
class IRSchedulerConfig:
    policy: SchedulerPolicy = SchedulerPolicy.STATIC
    max_batch: int = 1
    max_ctx: int = 2048
    timeout_ms: int = 30000


@dataclass
class IRTargetConfig:
    device: DeviceTarget = DeviceTarget.AUTO
    backend: str = ""
    memory_limit: str = ""


@dataclass
class IRModelConfig:
    name: str = ""
    weights: IRWeightsConfig = field(default_factory=IRWeightsConfig)
    cache: IRCacheConfig = field(default_factory=IRCacheConfig)
    attention: IRAttentionConfig = field(default_factory=IRAttentionConfig)
    decode: IRDecodeConfig = field(default_factory=IRDecodeConfig)
    scheduler: IRSchedulerConfig = field(default_factory=IRSchedulerConfig)
    target: IRTargetConfig = field(default_factory=IRTargetConfig)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'weights': {'quant_type': self.weights.quant_type.value, 'strategy': self.weights.strategy},
            'cache': {'strategy': self.cache.strategy.value, 'bits': self.cache.bits, 'max_size': self.cache.max_size},
            'attention': {'variant': self.attention.variant.value, 'streaming': self.attention.streaming, 'window_size': self.attention.window_size},
            'decode': {'mode': self.decode.mode.value, 'drafter': self.decode.drafter, 'k': self.decode.k, 'temperature': self.decode.temperature},
            'scheduler': {'policy': self.scheduler.policy.value, 'max_batch': self.scheduler.max_batch, 'max_ctx': self.scheduler.max_ctx},
            'target': {'device': self.target.device.value, 'backend': self.target.backend},
        }


@dataclass
class IRProgram:
    models: list[IRModelConfig] = field(default_factory=list)


class IRGenerator:
    """Converts TurboLang AST to TurboIR."""

    def generate(self, program: Program) -> IRProgram:
        ir_program = IRProgram()
        for model_ast in program.models:
            ir_program.models.append(self._lower_model(model_ast))
        return ir_program

    def _lower_model(self, model: ModelBlock) -> IRModelConfig:
        cfg = IRModelConfig(name=model.name)

        for directive in model.directives:
            if directive.keyword == 'weights':
                cfg.weights = self._lower_weights(directive)
            elif directive.keyword == 'kv_cache':
                cfg.cache = self._lower_cache(directive)
            elif directive.keyword == 'attention':
                cfg.attention = self._lower_attention(directive)
            elif directive.keyword == 'decode':
                cfg.decode = self._lower_decode(directive)
            elif directive.keyword == 'scheduler':
                cfg.scheduler = self._lower_scheduler(directive)
            elif directive.keyword == 'target':
                cfg.target = self._lower_target(directive)

        return cfg

    def _lower_weights(self, d) -> IRWeightsConfig:
        cfg = IRWeightsConfig()
        if len(d.positional) >= 1:
            cfg.strategy = d.positional[0]
        if len(d.positional) >= 2:
            dtype = d.positional[1]
            cfg.quant_type = QuantType(dtype) if dtype in [e.value for e in QuantType] else QuantType.FP16
        return cfg

    def _lower_cache(self, d) -> IRCacheConfig:
        cfg = IRCacheConfig()
        if d.positional:
            val = d.positional[0]
            cfg.strategy = CacheStrategy(val) if val in [e.value for e in CacheStrategy] else CacheStrategy.STANDARD
        cfg.bits = d.named.get('bits', 16)
        cfg.max_size = d.named.get('max_size', 0)
        return cfg

    def _lower_attention(self, d) -> IRAttentionConfig:
        cfg = IRAttentionConfig()
        if d.positional:
            val = d.positional[0]
            cfg.variant = AttentionVariant(val) if val in [e.value for e in AttentionVariant] else AttentionVariant.STANDARD
        cfg.streaming = d.named.get('streaming', False)
        cfg.window_size = d.named.get('window', 0)
        return cfg

    def _lower_decode(self, d) -> IRDecodeConfig:
        cfg = IRDecodeConfig()
        if d.positional:
            val = d.positional[0]
            cfg.mode = DecodeMode(val) if val in [e.value for e in DecodeMode] else DecodeMode.AUTOREGRESSIVE
        cfg.drafter = d.named.get('drafter', '')
        cfg.k = d.named.get('k', 5)
        cfg.temperature = d.named.get('temperature', 1.0)
        return cfg

    def _lower_scheduler(self, d) -> IRSchedulerConfig:
        cfg = IRSchedulerConfig()
        if d.positional:
            val = d.positional[0]
            cfg.policy = SchedulerPolicy(val) if val in [e.value for e in SchedulerPolicy] else SchedulerPolicy.STATIC
        cfg.max_batch = d.named.get('max_batch', 1)
        cfg.max_ctx = d.named.get('max_ctx', 2048)
        cfg.timeout_ms = d.named.get('timeout_ms', 30000)
        return cfg

    def _lower_target(self, d) -> IRTargetConfig:
        cfg = IRTargetConfig()
        if d.positional:
            val = d.positional[0]
            cfg.device = DeviceTarget(val) if val in [e.value for e in DeviceTarget] else DeviceTarget.AUTO
        if len(d.positional) >= 2:
            cfg.backend = d.positional[1]
        cfg.memory_limit = d.named.get('memory_limit', '')
        return cfg
