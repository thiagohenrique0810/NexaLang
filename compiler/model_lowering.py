"""Llama/NexaLM stateless prefill and serial activation lifetimes.

This lowering describes one sequence, no KV-cache side effects or in-place ops.
Weights remain external constants. Their residency, packed staging and kernel
workspace are separate requests supplied by the executor, never hidden here.
"""
from collections.abc import Mapping

from .model_config import ModelConfig
from .model_ir import DType, ModelGraph, ModelOp, TensorDesc, _WEIGHT_STORAGE, _integer
from .planner.memory import MemoryRequest


def lower_model(config: ModelConfig, sequence_length: int, *, weight_storage=None) -> ModelGraph:
    """Lower the configured Transformer to a complete, validated prefill graph.

    Optional weight_storage overrides may be partial. Every descriptor must
    identify a physical HF weight, preserve its shape/logical F32 type, and use
    F32 or Q4 matrix storage. Norm vectors remain F32. A tied lm_head reuses the
    embedding constant directly; it never creates a second weight allocation.
    """
    if not isinstance(config, ModelConfig):
        raise ValueError("config must be a ModelConfig")
    _integer(sequence_length, "sequence_length", 1)
    if sequence_length > config.max_position_embeddings:
        raise ValueError("sequence_length exceeds max_position_embeddings")
    if config.vocab_size > 2 ** 32:
        raise ValueError("vocab_size exceeds the U32 token ID range")
    shapes = config.required_tensor_shapes()
    storage = {} if weight_storage is None else weight_storage
    if not isinstance(storage, Mapping) or set(storage) - set(shapes):
        raise ValueError("weight_storage must map physical weight names to TensorDesc overrides")
    tensors = []
    for name, shape in shapes.items():
        tensor = storage.get(name, TensorDesc(name, shape))
        if (not isinstance(tensor, TensorDesc) or tensor.name != name or tensor.shape != shape
                or tensor.logical_dtype != DType.F32):
            raise ValueError(f"weight_storage descriptor does not match {name}")
        # A única lista de storages admitidos vive em model_ir; repeti-la aqui
        # foi o que deixou Q2/Q3/Q8 passarem na descrição e falharem no grafo.
        allowed = _WEIGHT_STORAGE if len(shape) == 2 else frozenset({DType.F32})
        if tensor.storage_dtype not in allowed:
            raise ValueError(f"unsupported physical storage for {name}")
        tensors.append(tensor)
    constants = tuple(shapes)
    tensors.append(TensorDesc("tokens", (sequence_length,), DType.U32, DType.U32))
    ops = []

    def emit(name, kind, inputs, width, attributes=None):
        tensors.append(TensorDesc(name, (sequence_length, width)))
        ops.append(ModelOp(name, kind, inputs, [name], {} if attributes is None else attributes))
        return name

    def project(name, source, weight, width):
        return emit(name, "MatMul", [source, weight], width, {"transpose_b": True})

    hidden = emit("embedding", "Embedding", ["tokens", "model.embed_tokens.weight"], config.hidden_size)
    kv_width = config.num_key_value_heads * config.head_dim
    for index in range(config.num_hidden_layers):
        prefix, weights = f"layers.{index}.", f"model.layers.{index}."
        normalized = emit(prefix + "input_norm", "RMSNorm", [hidden, weights + "input_layernorm.weight"],
                          config.hidden_size, {"epsilon": config.rms_norm_eps})
        query = project(prefix + "q", normalized, weights + "self_attn.q_proj.weight", config.hidden_size)
        key = project(prefix + "k", normalized, weights + "self_attn.k_proj.weight", kv_width)
        value = project(prefix + "v", normalized, weights + "self_attn.v_proj.weight", kv_width)
        query = emit(prefix + "q_rope", "RoPE", [query], config.hidden_size,
                     {"num_heads": config.num_attention_heads, "head_dim": config.head_dim, "theta": config.rope_theta})
        key = emit(prefix + "k_rope", "RoPE", [key], kv_width,
                   {"num_heads": config.num_key_value_heads, "head_dim": config.head_dim, "theta": config.rope_theta})
        attended = emit(prefix + "attention", "CausalAttention", [query, key, value], config.hidden_size,
                        {"num_heads": config.num_attention_heads,
                         "num_key_value_heads": config.num_key_value_heads, "head_dim": config.head_dim})
        projected = project(prefix + "attention_output", attended, weights + "self_attn.o_proj.weight", config.hidden_size)
        residual = emit(prefix + "attention_residual", "Add", [hidden, projected], config.hidden_size)
        normalized = emit(prefix + "post_attention_norm", "RMSNorm",
                          [residual, weights + "post_attention_layernorm.weight"], config.hidden_size,
                          {"epsilon": config.rms_norm_eps})
        gate = project(prefix + "gate", normalized, weights + "mlp.gate_proj.weight", config.intermediate_size)
        up = project(prefix + "up", normalized, weights + "mlp.up_proj.weight", config.intermediate_size)
        gated = emit(prefix + "swiglu", "SwiGLU", [gate, up], config.intermediate_size)
        down = project(prefix + "down", gated, weights + "mlp.down_proj.weight", config.hidden_size)
        hidden = emit(prefix + "output", "Add", [residual, down], config.hidden_size)
    normalized = emit("final_norm", "RMSNorm", [hidden, "model.norm.weight"], config.hidden_size,
                      {"epsilon": config.rms_norm_eps})
    head = config.tensor_aliases().get("lm_head.weight", "lm_head.weight")
    project("logits", normalized, head, config.vocab_size)
    return ModelGraph(name=f"{config.name}.prefill", tensors=tensors, ops=ops,
                      inputs=["tokens"], outputs=["logits"], constants=constants)


def derive_activation_requests(graph: ModelGraph) -> list[MemoryRequest]:
    """Derive half-open lifetimes from the graph's serial operation order.

    Inputs exist at event 0. Operation i reads and writes at event i+1, so both
    its operands and result live through that event; the planner cannot alias
    them. Every last use extends end to the following event. Graph outputs are
    retained through the final consumer event len(ops)+1. Constants are excluded.
    Unused values still occupy storage for their input/production event.
    """
    if not isinstance(graph, ModelGraph):
        raise ValueError("graph must be a ModelGraph")
    graph.validate()
    constants = set(graph.constants)
    starts = {name: 0 for name in graph.inputs}
    ends = {name: 1 for name in graph.inputs}
    for index, op in enumerate(graph.ops, 1):
        for name in op.inputs:
            if name not in constants:
                ends[name] = max(ends[name], index + 1)
        for name in op.outputs:
            starts[name] = index
            ends[name] = index + 1
    for name in graph.outputs:
        if name not in constants:
            ends[name] = len(graph.ops) + 2
    return [MemoryRequest(tensor.name, tensor.storage_nbytes, starts[tensor.name], ends[tensor.name],
                          alignment=tensor.alignment, tier=tensor.tier.value)
            for tensor in graph.tensors if tensor.name not in constants]
