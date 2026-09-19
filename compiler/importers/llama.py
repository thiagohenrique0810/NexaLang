"""Import validated local bias-free Llama checkpoints into NexaPack bundles.

No model code is imported or executed, and tokenizer assets are preserved as
opaque local files. This importer does not provide tokenization or inference.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct

from compiler.model_config import ModelConfig
from runtime.nexapack.bundle import write_model_bundle
from .safetensors import (
    READ_CHUNK_BYTES, SafeTensorCheckpoint, SafeTensorError,
    _signature, file_sha256, read_json, safe_child,
)

_ASSETS = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
           'special_tokens_map.json', 'tokenizer.model')


def _decoded_digest(checkpoint, name):
    """Hash canonical decoded float32 values with one open shard at a time."""
    digest = hashlib.sha256()
    scratch = bytearray(READ_CHUNK_BYTES)
    view = memoryview(scratch)
    used = 0
    for row in checkpoint.iter_rows(name):
        for value in row:
            if not math.isfinite(value):
                raise SafeTensorError(f'Nonfinite value in tied tensor: {name}')
            # Positive/negative zero are equal mathematical weights. Other
            # decoded values must have identical float32 representations.
            struct.pack_into('<f', scratch, used, value if value else 0.0)
            used += 4
            if used == len(scratch):
                digest.update(view)
                used = 0
    digest.update(view[:used])
    return digest.digest()


def import_llama_checkpoint(source_dir, destination, *, group_size=32, block_rows=64):
    """Validate an entire local checkpoint schema before publishing a bundle.

    Tied lm_head weights may be absent. If present, their decoded float32 values
    must match the embeddings before the redundant tensor is omitted. Unknown
    tensors are rejected, so unsupported biases/features cannot silently vanish.
    """
    source = Path(source_dir).resolve(strict=True)
    config_path = safe_child(source, 'config.json')
    config_signature = _signature(config_path.stat())
    raw_config = read_json(config_path)
    if _signature(config_path.stat()) != config_signature:
        raise SafeTensorError('config.json changed while reading')
    if not isinstance(raw_config, dict):
        raise SafeTensorError('config.json must contain an object')
    config = ModelConfig.from_hf_config(raw_config)
    checkpoint = SafeTensorCheckpoint(source)
    required = config.required_tensor_shapes()
    aliases = config.tensor_aliases()
    present = checkpoint.tensor_shapes
    missing = set(required) - set(present)
    extra = set(present) - set(required) - set(aliases)
    if missing or extra:
        details = []
        if missing:
            details.append('missing tensors: ' + ', '.join(sorted(missing)))
        if extra:
            details.append('unsupported tensors: ' + ', '.join(sorted(extra)))
        raise SafeTensorError('; '.join(details))
    for name, shape in required.items():
        if present[name] != tuple(shape):
            raise SafeTensorError(f'Tensor shape mismatch for {name}: {present[name]} != {tuple(shape)}')
    confirmed_aliases = []
    for alias, target in aliases.items():
        if alias in present:
            if present[alias] != tuple(required[target]):
                raise SafeTensorError(f'Tied tensor shape mismatch: {alias}')
            if _decoded_digest(checkpoint, alias) != _decoded_digest(checkpoint, target):
                raise SafeTensorError(f'Tied tensor differs from its shared weights: {alias} != {target}')
            confirmed_aliases.append(alias)
    assets = {}
    for name in _ASSETS:
        if (source / name).exists():
            assets[name] = safe_child(source, name)
    provenance = checkpoint.provenance(hash_files=True)
    provenance['assets'] = []
    for name, path in sorted(assets.items()):
        signature = _signature(path.stat())
        digest = file_sha256(path)
        if _signature(path.stat()) != signature:
            raise SafeTensorError(f'Asset changed while computing provenance: {name}')
        provenance['assets'].append({'path': name, 'size_bytes': signature[2], 'sha256': digest})
    if _signature(config_path.stat()) != config_signature:
        raise SafeTensorError('config.json changed after its configuration was validated')
    provenance['tied_weights_verified'] = confirmed_aliases
    sources = {name: (lambda name=name: checkpoint.iter_rows(name)) for name in required}
    asset_checksums = {entry['path']: entry['sha256'] for entry in provenance['assets']}
    provenance['source_config'] = {'path': 'config.json', 'sha256': asset_checksums['config.json']}
    write_model_bundle(destination, config, sources, group_size=group_size,
                       block_rows=block_rows, tokenizer_files=assets, provenance=provenance,
                       asset_checksums=asset_checksums)
    return {
        'format': 'NexaModelBundle', 'format_version': 1, 'source_format': 'safetensors',
        'destination': str(Path(destination).resolve()), 'config': config.to_dict(),
        'tensor_count': len(required), 'parameter_count': config.parameter_count(),
        'tensor_aliases': aliases, 'tied_weights_verified': confirmed_aliases,
        'assets': sorted(assets), 'provenance': provenance,
    }
