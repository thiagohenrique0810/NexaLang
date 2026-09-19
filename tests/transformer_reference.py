"""Independent PyTorch Llama oracle for small offline numerical diagnostics.

Equations follow the official Hugging Face Llama implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
This functional implementation does not call NexaLang native compute kernels.
Q4 weights retain exact scale*code products in float64 matmuls; activation
boundaries round to float32, matching the runtime's documented arithmetic.

Unlike the runtime, this diagnostic materializes weights and attention scores.
Its allocations are outside the runtime budget. Limits prevent accidental use
as a full-model runner. PyTorch is imported only when the oracle is requested.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_REFERENCE_PARAMETERS = 2_000_000
MAX_REFERENCE_SEQUENCE = 256


def torch_available():
    return importlib.util.find_spec('torch') is not None


@lru_cache(maxsize=1)
def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('PyTorch is required only for optional reference verification') from error
    return torch


def _check_size(config):
    if config.parameter_count() > MAX_REFERENCE_PARAMETERS:
        raise ValueError(f'Reference verification is limited to {MAX_REFERENCE_PARAMETERS} parameters')


def _kv_options(codec, group_size, bits=None, seed=None, codebook_f32le=None):
    # A group size alone selected Q4 in the original diagnostic API.
    codec = ('q4' if group_size is not None else 'f32') if codec is None else codec
    if codec not in ('f32', 'q4', 'q3', 'tq'):
        raise ValueError('Reference KV codec must be f32, q4, q3 or tq')
    if codec != 'tq' and any(value is not None for value in (bits, seed, codebook_f32le)):
        raise ValueError('TQ KV parameters require the tq codec')
    if codec in ('f32', 'tq'):
        if group_size is not None:
            raise ValueError('F32/TQ KV does not accept a group size')
    else:
        from runtime.nexapack.format import MAX_GROUP_SIZE
        group_size = 32 if group_size is None else group_size
        if type(group_size) is not int or not 0 < group_size <= MAX_GROUP_SIZE:
            raise ValueError(f'KV group size must be a positive integer up to {MAX_GROUP_SIZE}')
    return codec, group_size


def load_bundle_weights(bundle_path):
    """Materialize a small bundle as exact decoded-Q4 float64 reference tensors."""
    from runtime.nexapack.bundle import ModelBundleReader
    from runtime.nexapack.format import decode_q4_row
    torch = _torch()
    weights = {}
    with ModelBundleReader(bundle_path) as bundle:
        config = bundle.config
        _check_size(config)
        for name, shape in config.required_tensor_shapes().items():
            if len(shape) == 1:
                weights[name] = torch.tensor(bundle.read_f32(name), dtype=torch.float64)
            else:
                values = []
                with bundle.open_q4(name) as reader:
                    for row in range(reader.rows):
                        values.append(decode_q4_row(reader.read_rows(row, 1), reader.cols, reader.group_size))
                weights[name] = torch.tensor(values, dtype=torch.float64)
        for alias, target in config.tensor_aliases().items():
            weights[alias] = weights[target]
    return config, weights


def load_safetensors_weights(source_dir, *, expected_provenance=None):
    """Materialize supported original checkpoint weights for quantization error."""
    from compiler.importers.safetensors import SafeTensorCheckpoint, file_sha256, read_json, safe_child
    from compiler.model_config import ModelConfig
    torch = _torch()
    config_path = safe_child(source_dir, 'config.json')
    config = ModelConfig.from_hf_config(read_json(config_path))
    _check_size(config)
    checkpoint = SafeTensorCheckpoint(source_dir)
    if expected_provenance is not None:
        if (not isinstance(expected_provenance, dict)
                or expected_provenance.get('format') != 'safetensors'
                or expected_provenance.get('files') != checkpoint.provenance()['files']
                or expected_provenance.get('source_config') != {
                    'path': 'config.json', 'sha256': file_sha256(config_path)}):
            raise ValueError('Reference checkpoint does not match the bundle source provenance')
    required, aliases = config.required_tensor_shapes(), config.tensor_aliases()
    if set(required) - set(checkpoint.tensor_shapes) or set(checkpoint.tensor_shapes) - set(required) - set(aliases):
        raise ValueError('Reference checkpoint has missing or unsupported tensors')
    weights = {}
    for name, shape in required.items():
        if checkpoint.tensor_shapes[name] != shape:
            raise ValueError(f'Reference checkpoint shape mismatch: {name}')
        rows = [list(row) for row in checkpoint.iter_rows(name)]
        weights[name] = torch.tensor(rows[0] if len(shape) == 1 else rows, dtype=torch.float64)
    for alias, target in aliases.items():
        if alias in checkpoint.tensor_shapes:
            values = torch.tensor([list(row) for row in checkpoint.iter_rows(alias)], dtype=torch.float64)
            if not torch.equal(values, weights[target]):
                raise ValueError(f'Reference checkpoint tied weights differ: {alias}')
        weights[alias] = weights[target]
    return config, weights


class TorchLlamaReference:
    """Functional Llama with incremental F32 or headwise decoded-Q4/Q3/TQ KV.

    Quantization affects new cache rows immediately, including the current position.
    Existing cache values are never quantized again when decoding a new token.
    """
    def __init__(self, config, weights, *, kv_group_size=None, kv_codec=None,
                 kv_bits=None, kv_seed=None, kv_codebook_f32le=None):
        torch = _torch()
        _check_size(config)
        self.config = config
        self.kv_codec, self.kv_group_size = _kv_options(
            kv_codec, kv_group_size, kv_bits, kv_seed, kv_codebook_f32le)
        self._tq = None
        if self.kv_codec == 'tq':
            from tq_reference import TQReference
            self._tq = TQReference(config.head_dim, 3 if kv_bits is None else kv_bits,
                                   42 if kv_seed is None else kv_seed, kv_codebook_f32le)
        self.weights = {}
        for name, shape in config.required_tensor_shapes().items():
            if name not in weights:
                raise ValueError(f'Missing reference weight: {name}')
            tensor = torch.as_tensor(weights[name], dtype=torch.float64, device='cpu').detach()
            if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Invalid reference weight: {name}')
            self.weights[name] = tensor
        for alias, target in config.tensor_aliases().items():
            self.weights[alias] = self.weights[target]
        self.reset()

    def reset(self):
        self.token_ids = ()
        self._cache = None

    def _tokens(self, token_ids, previous=0):
        try:
            tokens = tuple(token_ids)
        except TypeError as error:
            raise ValueError('Token IDs must be an iterable of integers') from error
        if not tokens or any(type(token) is not int or not 0 <= token < self.config.vocab_size for token in tokens):
            raise ValueError('Token IDs must be nonempty valid vocabulary integers')
        if previous + len(tokens) > min(self.config.max_position_embeddings, MAX_REFERENCE_SEQUENCE):
            raise ValueError('Sequence exceeds reference context capacity')
        return tokens

    def _linear(self, x, name):
        return (x.double() @ self.weights[name].T).float()

    def _norm(self, x, name):
        x = x.double()
        return (x * (x.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps).rsqrt()
                * self.weights[name]).float()

    def _rope(self, x, start):
        torch = _torch()
        head_dim = self.config.head_dim
        frequencies = self.config.rope_theta ** (-2 * torch.arange(head_dim // 2, dtype=torch.float64) / head_dim)
        angles = torch.arange(start, start + x.shape[0], dtype=torch.float64)[:, None] * frequencies[None, :]
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
        left, right = x.double().split(head_dim // 2, dim=-1)
        return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1).float()

    def _cache_values(self, key, value):
        if self.kv_codec == 'f32':
            return key, value
        if self.kv_codec == 'tq':
            torch = _torch()
            decoded = []
            for tensor in (key, value):
                rows = [self._tq.decode(self._tq.encode(row.tolist()))
                        for row in tensor.reshape(-1, self.config.head_dim)]
                # TQ reconstruction rounds to F32 before double attention reduction.
                decoded.append(torch.tensor(rows, dtype=torch.float32).reshape(tensor.shape))
            return tuple(decoded)
        if self.kv_codec == 'q3':
            from q3_reference import decode_q3_row as decode, quantize_q3_row as quantize
        else:
            from runtime.nexapack.format import decode_q4_row as decode, quantize_q4_row as quantize
        torch = _torch()
        width = self.config.head_dim
        decoded = []
        for tensor in (key, value):
            rows = [decode(quantize(row.tolist(), self.kv_group_size), width, self.kv_group_size)
                    for row in tensor.reshape(-1, width)]
            # The packed attention kernel consumes exact scale*code products;
            # rounding these rows back to F32 would define a different oracle.
            decoded.append(torch.tensor(rows, dtype=torch.float64).reshape(tensor.shape))
        return tuple(decoded)

    def _forward(self, tokens, start, old_cache):
        torch = _torch()
        config = self.config
        count, heads, kv_heads, width = len(tokens), config.num_attention_heads, config.num_key_value_heads, config.head_dim
        x = self.weights['model.embed_tokens.weight'][list(tokens)].float()
        new_cache = []
        with torch.no_grad():
            for layer in range(config.num_hidden_layers):
                prefix = f'model.layers.{layer}.'
                normalized = self._norm(x, prefix + 'input_layernorm.weight')
                q = self._rope(self._linear(normalized, prefix + 'self_attn.q_proj.weight').reshape(count, heads, width), start)
                k = self._rope(self._linear(normalized, prefix + 'self_attn.k_proj.weight').reshape(count, kv_heads, width), start)
                v = self._linear(normalized, prefix + 'self_attn.v_proj.weight').reshape(count, kv_heads, width)
                k, v = self._cache_values(k, v)
                if old_cache is not None:
                    k = torch.cat((old_cache[layer][0], k), dim=0)
                    v = torch.cat((old_cache[layer][1], v), dim=0)
                new_cache.append((k, v))
                expanded_k = k.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).double()
                expanded_v = v.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).double()
                scores = q.transpose(0, 1).double() @ expanded_k.transpose(1, 2) / (width ** 0.5)
                query_positions = torch.arange(start, start + count)
                key_positions = torch.arange(k.shape[0])
                scores.masked_fill_(key_positions[None, :] > query_positions[:, None], -torch.inf)
                # Native scratch stores the unnormalized exponentials as f32;
                # their sum and weighted-V reduction remain double precision.
                exponentials = (scores - scores.amax(dim=-1, keepdim=True)).exp().float().double()
                attention = (exponentials @ expanded_v / exponentials.sum(dim=-1, keepdim=True)).float()
                attention = attention.transpose(0, 1).contiguous().reshape(count, config.hidden_size)
                x = x + self._linear(attention, prefix + 'self_attn.o_proj.weight')
                normalized = self._norm(x, prefix + 'post_attention_layernorm.weight')
                gate = self._linear(normalized, prefix + 'mlp.gate_proj.weight').double()
                up = self._linear(normalized, prefix + 'mlp.up_proj.weight').double()
                gated = (gate * gate.sigmoid() * up).float()
                x = x + self._linear(gated, prefix + 'mlp.down_proj.weight')
            logits = self._linear(self._norm(x, 'model.norm.weight'), 'lm_head.weight')
        if not bool(torch.isfinite(logits).all()):
            raise ArithmeticError('Reference forward produced nonfinite logits')
        return logits, new_cache

    def prefill(self, token_ids):
        tokens = self._tokens(token_ids)
        logits, cache = self._forward(tokens, 0, None)
        self.token_ids, self._cache = tokens, cache
        return logits

    def decode(self, token_id):
        if not self.token_ids:
            raise ValueError('Decode requires a prior successful prefill')
        token, = self._tokens((token_id,), len(self.token_ids))
        logits, cache = self._forward((token,), len(self.token_ids), self._cache)
        self.token_ids, self._cache = self.token_ids + (token,), cache
        return logits[0]


def llama_forward(config, weights, token_ids, *, kv_group_size=None, kv_codec=None,
                  kv_bits=None, kv_seed=None, kv_codebook_f32le=None):
    return TorchLlamaReference(config, weights, kv_group_size=kv_group_size,
                               kv_codec=kv_codec, kv_bits=kv_bits, kv_seed=kv_seed,
                               kv_codebook_f32le=kv_codebook_f32le).prefill(token_ids)


def compare_logits(actual, expected, *, atol=1e-5, rtol=1e-4):
    torch = _torch()
    actual, expected = torch.as_tensor(actual, dtype=torch.float64), torch.as_tensor(expected, dtype=torch.float64)
    if actual.shape != expected.shape or not actual.numel():
        raise ValueError(f'Logit shapes differ or are empty: {tuple(actual.shape)} vs {tuple(expected.shape)}')
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError('Logits must be finite')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError('Tolerances must be finite and nonnegative')
    errors = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    return {'passed': bool((errors <= allowed).all()), 'atol': atol, 'rtol': rtol,
            'max_abs_error': errors.max().item(),
            'mean_abs_error': errors.mean().item(),
            'max_rel_error': (errors / expected.abs().clamp_min(1e-12)).max().item()}


def verify_bundle_forward(bundle_path, token_ids, actual_logits, *, source_dir=None,
                          kv_group_size=None, kv_codec=None, kv_bits=None, kv_seed=None,
                          kv_codebook_f32le=None, atol=1e-5, rtol=1e-4):
    """Separate native, weight-quantization and optional KV-quantization errors.

    quantization_error always retains its original meaning: Q4 weights/F32 KV
    versus source weights/F32 KV. Packed KV introduces a separate comparison against
    the same Q4 weights with F32 KV, plus combined error when a source is supplied.
    """
    kv_codec, kv_group_size = _kv_options(kv_codec, kv_group_size, kv_bits, kv_seed, kv_codebook_f32le)
    config, weights = load_bundle_weights(bundle_path)
    quantized_logits = llama_forward(config, weights, token_ids)
    execution_reference = quantized_logits if kv_codec == 'f32' else llama_forward(
        config, weights, token_ids, kv_group_size=kv_group_size, kv_codec=kv_codec,
        kv_bits=kv_bits, kv_seed=kv_seed, kv_codebook_f32le=kv_codebook_f32le)
    execution = compare_logits(actual_logits, execution_reference, atol=atol, rtol=rtol)
    def representation_error(actual, expected):
        error = compare_logits(actual, expected, atol=atol, rtol=rtol)
        return {key: value for key, value in error.items() if key not in ('passed', 'atol', 'rtol')}
    quantization = None
    combined = None
    if source_dir is not None:
        from runtime.nexapack.bundle import ModelBundleReader
        with ModelBundleReader(bundle_path) as bundle:
            provenance = bundle.manifest['provenance']['data']
        original_config, original_weights = load_safetensors_weights(source_dir, expected_provenance=provenance)
        if original_config.to_dict() != config.to_dict():
            raise ValueError('Reference checkpoint config does not match the bundle')
        original_logits = llama_forward(original_config, original_weights, token_ids)
        quantization = representation_error(quantized_logits, original_logits)
        if kv_codec != 'f32':
            combined = representation_error(execution_reference, original_logits)
    report = {'reference': 'pytorch_decoded_q4', 'verified': execution['passed'],
              'execution_error': execution, 'quantization_error': quantization,
              'scope': 'diagnostic allocations outside the native runtime memory budget',
              'quality_or_perplexity_measured': False}
    if kv_codec != 'f32':
        report.update({'reference': f'pytorch_decoded_q4_kv_{kv_codec}',
                       'kv_quantization_error': representation_error(execution_reference, quantized_logits)})
        if kv_codec == 'tq':
            from tq_reference import TRANSFORM_ID
            report.update({'kv_bits': 3 if kv_bits is None else kv_bits,
                           'kv_seed': 42 if kv_seed is None else kv_seed,
                           'kv_codebook_f32le': kv_codebook_f32le, 'kv_transform_id': TRANSFORM_ID})
        else:
            report['kv_group_size'] = kv_group_size
        if combined is not None:
            report['combined_quantization_error'] = combined
    return report
