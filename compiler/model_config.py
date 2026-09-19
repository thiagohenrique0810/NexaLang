"""Physical tensor contract for bias-free Llama-compatible NexaLM models.

The supported numerical architecture uses full-head, unscaled RoPE, RMSNorm,
causal GQA attention and SwiGLU. Tokenizer and generation settings are separate.
The Hugging Face adapter accepts a deliberately bounded configuration subset;
it never silently ignores unknown architectural extensions.
"""
from dataclasses import dataclass, fields
import json
import math

from .model_ir import _integer, _keys, _load_json, _name


MAX_MODEL_TENSORS = 4096


def _positive_float(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


_HF_FIELDS = {
    "name", "model_type", "vocab_size", "hidden_size", "intermediate_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "max_position_embeddings", "rope_theta", "rms_norm_eps", "tie_word_embeddings",
    "architectures", "hidden_act", "attention_bias", "mlp_bias", "rope_scaling",
    "rope_parameters", "head_dim", "partial_rotary_factor", "rope_interleaved",
    "sliding_window", "use_sliding_window", "quantization_config", "pretraining_tp",
    "is_encoder_decoder", "add_cross_attention", "cross_attention_hidden_size",
    "tie_encoder_decoder", "pruned_heads", "auto_map",
}
# These affect checkpoint provenance, training or inference API behavior, not
# the supported evaluation-time tensor equations. They are not copied to the
# model contract or used to select an implementation.
_HF_METADATA = {
    "_name_or_path", "_commit_hash", "_attn_implementation", "_attn_implementation_internal",
    "_flash_attn_2_enabled", "transformers_version", "torch_dtype", "dtype",
    "bos_token_id", "eos_token_id", "pad_token_id", "sep_token_id", "decoder_start_token_id",
    "attention_dropout", "initializer_range", "use_cache", "return_dict", "output_attentions",
    "output_hidden_states", "torchscript", "is_decoder", "task_specific_params",
    "finetuning_task", "id2label", "label2id", "num_labels", "problem_type",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    def __post_init__(self):
        _name(self.name, "model name")
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "max_position_embeddings"):
            _integer(getattr(self, name), name, 1)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if type(self.tie_word_embeddings) is not bool:
            raise ValueError("tie_word_embeddings must be a boolean")
        object.__setattr__(self, "rope_theta", _positive_float(self.rope_theta, "rope_theta"))
        object.__setattr__(self, "rms_norm_eps", _positive_float(self.rms_norm_eps, "rms_norm_eps"))

    @property
    def architecture(self):
        """Numerical architecture family; NexaLM uses the same tensor equations."""
        return "llama"

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def required_tensor_shapes(self):
        """Physical weight tensors in HF order [output_features, input_features].

        Aliases are deliberately excluded: a tied output projection occupies
        no additional physical storage. RoPE tables and KV cache are runtime
        state, not learned parameters. The bundle contract permits at most
        MAX_MODEL_TENSORS physical tensors; reject before expanding layer names.
        """
        tensor_count = 2 + 9 * self.num_hidden_layers + (not self.tie_word_embeddings)
        if tensor_count > MAX_MODEL_TENSORS:
            raise ValueError(f"model requires {tensor_count} physical tensors; limit is {MAX_MODEL_TENSORS}")
        hidden, intermediate = self.hidden_size, self.intermediate_size
        kv_width = self.num_key_value_heads * self.head_dim
        result = {"model.embed_tokens.weight": (self.vocab_size, hidden)}
        for layer in range(self.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            result.update({
                prefix + "input_layernorm.weight": (hidden,),
                prefix + "self_attn.q_proj.weight": (hidden, hidden),
                prefix + "self_attn.k_proj.weight": (kv_width, hidden),
                prefix + "self_attn.v_proj.weight": (kv_width, hidden),
                prefix + "self_attn.o_proj.weight": (hidden, hidden),
                prefix + "post_attention_layernorm.weight": (hidden,),
                prefix + "mlp.gate_proj.weight": (intermediate, hidden),
                prefix + "mlp.up_proj.weight": (intermediate, hidden),
                prefix + "mlp.down_proj.weight": (hidden, intermediate),
            })
        result["model.norm.weight"] = (hidden,)
        if not self.tie_word_embeddings:
            result["lm_head.weight"] = (self.vocab_size, hidden)
        return result

    def tensor_aliases(self):
        return {"lm_head.weight": "model.embed_tokens.weight"} if self.tie_word_embeddings else {}

    def parameter_count(self):
        """Exact learned scalar count, counting shared embeddings once."""
        hidden, kv_width = self.hidden_size, self.num_key_value_heads * self.head_dim
        per_layer = (2 * hidden * hidden + 2 * hidden * kv_width
                     + 3 * hidden * self.intermediate_size + 2 * hidden)
        embeddings = self.vocab_size * hidden * (1 if self.tie_word_embeddings else 2)
        return embeddings + self.num_hidden_layers * per_layer + hidden

    def to_dict(self):
        return {"schema_version": 1, **{field.name: getattr(self, field.name) for field in fields(self)}}

    @classmethod
    def from_dict(cls, data):
        names = {field.name for field in fields(cls)}
        _keys(data, names | {"schema_version"}, "ModelConfig")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported ModelConfig schema_version")
        return cls(**{name: data[name] for name in names})

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(_load_json(text))

    @classmethod
    def from_hf_config(cls, data):
        """Import a supported HF config without transformers or remote code.

        Missing KV heads mean ordinary multi-head attention. Omitted tying and
        RMSNorm epsilon follow Llama defaults (False and 1e-6), rather than the
        NexaLM constructor defaults. Structural dimensions must be explicit.
        """
        if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
            raise ValueError("HF config must be an object with string keys")
        unknown = set(data) - _HF_FIELDS - _HF_METADATA
        if unknown:
            raise ValueError(f"unsupported HF config fields: {', '.join(sorted(unknown))}")
        if data.get("model_type") not in ("llama", "nexalm", "nexalm512"):
            raise ValueError("only Llama-compatible model_type llama/nexalm/nexalm512 is supported")
        architectures = data.get("architectures")
        supported = {"LlamaForCausalLM", "LlamaModel", "NexaLMForCausalLM", "NexaLM512ForCausalLM"}
        if architectures is not None and (not isinstance(architectures, list)
                or any(not isinstance(name, str) or name not in supported for name in architectures)):
            raise ValueError("unsupported HF architectures")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError("only hidden_act='silu' (SwiGLU) is supported")
        for name in ("attention_bias", "mlp_bias", "rope_interleaved", "use_sliding_window",
                     "is_encoder_decoder", "add_cross_attention", "tie_encoder_decoder"):
            if name in data and (type(data[name]) is not bool or data[name]):
                raise ValueError(f"{name} must be false for the supported architecture")
        for name in ("rope_scaling", "sliding_window", "quantization_config", "cross_attention_hidden_size"):
            if data.get(name) is not None:
                raise ValueError(f"{name} is not supported")
        for name in ("auto_map", "pruned_heads"):
            if name in data and data[name] not in (None, {}):
                raise ValueError(f"{name} is not supported")
        if "pretraining_tp" in data and data["pretraining_tp"] is not None:
            if type(data["pretraining_tp"]) is not int or data["pretraining_tp"] != 1:
                raise ValueError("only pretraining_tp=1 is supported")
        if "partial_rotary_factor" in data:
            if _positive_float(data["partial_rotary_factor"], "partial_rotary_factor") != 1:
                raise ValueError("partial RoPE is not supported")
        rope_theta = _positive_float(data.get("rope_theta", 10000.0), "rope_theta")
        rope_parameters = data.get("rope_parameters")
        if rope_parameters is not None:
            if (not isinstance(rope_parameters, dict)
                    or set(rope_parameters) - {"rope_type", "rope_theta"}
                    or rope_parameters.get("rope_type", "default") != "default"):
                raise ValueError("only default, unscaled rope_parameters are supported")
            parameter_theta = _positive_float(rope_parameters.get("rope_theta", rope_theta), "rope_theta")
            if "rope_theta" in data and parameter_theta != rope_theta:
                raise ValueError("rope_theta and rope_parameters disagree")
            rope_theta = parameter_theta
        required = {"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "max_position_embeddings"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"missing HF structural fields: {', '.join(sorted(missing))}")
        dimensions = {name: data[name] for name in required}
        kv_heads = data.get("num_key_value_heads")
        if "_name_or_path" in data and data["_name_or_path"] is not None and not isinstance(data["_name_or_path"], str):
            raise ValueError("_name_or_path must be a string or null")
        name = data["name"] if "name" in data else data.get("_name_or_path") or data["model_type"]
        config = cls(name=name,
                     **dimensions,
                     num_key_value_heads=dimensions["num_attention_heads"] if kv_heads is None else kv_heads,
                     rope_theta=rope_theta, rms_norm_eps=data.get("rms_norm_eps", 1e-6),
                     tie_word_embeddings=data.get("tie_word_embeddings", False))
        if data.get("head_dim") is not None:
            _integer(data["head_dim"], "head_dim", 1)
            if data["head_dim"] != config.head_dim:
                raise ValueError("head_dim must equal hidden_size / num_attention_heads")
        return config
