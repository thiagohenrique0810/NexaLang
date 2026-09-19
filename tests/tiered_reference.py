"""Independent chunk-aware PyTorch oracle for the CPU age-tier KV policy.

This diagnostic retains decoded cache tensors and uses Python Q4/Q3 codecs.
It never calls the tier planner, session, or native migration/attention kernels.
Its allocations are outside the execution budget and inherited size limits apply.
"""
from __future__ import annotations

from transformer_reference import (
    TorchLlamaReference, _torch, compare_logits, llama_forward,
    load_bundle_weights, load_safetensors_weights,
)


def _options(page_tokens, hot_pages, warm_pages, group_size):
    for name, value, minimum in (("page_tokens", page_tokens, 1), ("hot_pages", hot_pages, 1),
                                 ("warm_pages", warm_pages, 0), ("group_size", group_size, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"Reference {name} must be an integer >= {minimum}")
    if group_size > 1 << 20:
        raise ValueError("Reference group_size exceeds the portable codec limit")
    if hot_pages > 65536 or warm_pages > 65536:
        raise ValueError("Reference tier counts exceed the page metadata limit")


def page_codec(index, count, *, hot_pages, warm_pages):
    age = count - 1 - index
    return "f32" if age < hot_pages else "q4" if age < hot_pages + warm_pages else "q3"


class TorchTieredReference(TorchLlamaReference):
    """New rows are F32; complete-page migrations follow each successful chunk."""
    def __init__(self, config, weights, *, page_tokens, hot_pages=1, warm_pages=1, group_size=32):
        _options(page_tokens, hot_pages, warm_pages, group_size)
        self.page_tokens, self.hot_pages, self.warm_pages = page_tokens, hot_pages, warm_pages
        self.group_size = group_size
        super().__init__(config, weights)

    def reset(self):
        super().reset()
        self.page_codecs = ()
        self.packed_rows = {}

    def _migrate(self, cache, length, old_codecs, old_records):
        from q3_reference import decode_q3_row, quantize_q3_row
        from runtime.nexapack.format import decode_q4_row, quantize_q4_row
        torch = _torch()
        count = (length + self.page_tokens - 1) // self.page_tokens
        codecs = list(old_codecs) + ["f32"] * (count - len(old_codecs))
        records = dict(old_records)
        migrated = [(key.double().clone(), value.double().clone()) for key, value in cache]
        for page in range(count):
            begin = page * self.page_tokens
            end = min(length, begin + self.page_tokens)
            target = page_codec(page, count, hot_pages=self.hot_pages, warm_pages=self.warm_pages)
            if target == codecs[page]:
                continue
            if end - begin != self.page_tokens or target == "f32":
                raise ValueError("The age policy may only demote complete pages")
            quantize, decode = ((quantize_q4_row, decode_q4_row) if target == "q4"
                                else (quantize_q3_row, decode_q3_row))
            for layer, pair in enumerate(migrated):
                for kind, tensor in enumerate(pair):
                    for position in range(begin, end):
                        for head in range(self.config.num_key_value_heads):
                            # Q4->Q3 consumes reconstructed Q4 rounded to F32,
                            # never hidden original K/V coordinates.
                            bridge = tensor[position, head].float().tolist()
                            packed = quantize(bridge, self.group_size)
                            values = decode(packed, self.config.head_dim, self.group_size)
                            tensor[position, head] = torch.tensor(values, dtype=torch.float64)
                            records[layer, kind, position, head] = packed
            codecs[page] = target
        return migrated, tuple(codecs), records

    def _perform(self, token_ids, *, replace):
        previous = 0 if replace else len(self.token_ids)
        if not replace and not previous:
            raise ValueError("Append requires a prior successful prefill")
        tokens = self._tokens(token_ids, previous)
        logits, cache = self._forward(tokens, previous, None if replace else self._cache)
        cache, codecs, records = self._migrate(cache, previous + len(tokens),
                                               () if replace else self.page_codecs,
                                               {} if replace else self.packed_rows)
        self.token_ids = tokens if replace else self.token_ids + tokens
        self._cache, self.page_codecs, self.packed_rows = cache, codecs, records
        return logits

    def prefill(self, token_ids):
        return self._perform(token_ids, replace=True)

    def append(self, token_ids):
        return self._perform(token_ids, replace=False)

    def decode(self, token_id):
        return self.append([token_id])[0]


def verify_tiered_bundle_forward(bundle_path, token_chunks, actual_logits, *, page_tokens,
                                  hot_pages=1, warm_pages=1, group_size=32, source_dir=None,
                                  atol=1e-5, rtol=1e-4):
    """Replay exact chunk boundaries and distinguish execution/representation errors."""
    _options(page_tokens, hot_pages, warm_pages, group_size)
    if (not isinstance(token_chunks, (list, tuple)) or not token_chunks
            or any(not isinstance(chunk, (list, tuple)) or not chunk for chunk in token_chunks)):
        raise ValueError("Tier verification requires nonempty token chunks in execution order")
    config, weights = load_bundle_weights(bundle_path)
    oracle = TorchTieredReference(config, weights, page_tokens=page_tokens, hot_pages=hot_pages,
                                  warm_pages=warm_pages, group_size=group_size)
    chunks = [list(chunk) for chunk in token_chunks]
    expected = [oracle.prefill(chunks[0])]
    expected.extend(oracle.append(chunk) for chunk in chunks[1:])
    expected = _torch().cat(expected, dim=0)
    tokens = [token for chunk in chunks for token in chunk]
    quantized = llama_forward(config, weights, tokens)
    execution = compare_logits(actual_logits, expected, atol=atol, rtol=rtol)

    def representation_error(actual, reference):
        return {name: value for name, value in compare_logits(actual, reference, atol=atol, rtol=rtol).items()
                if name not in ("passed", "atol", "rtol")}

    report = {
        "reference": "pytorch_decoded_q4_kv_age_tiers", "verified": execution["passed"],
        "execution_error": execution, "quantization_error": None,
        "kv_quantization_error": representation_error(expected, quantized),
        "kv_policy": "age", "kv_page_tokens": page_tokens, "kv_hot_pages": hot_pages,
        "kv_warm_pages": warm_pages, "kv_group_size": group_size,
        "token_chunks": chunks, "final_page_codecs": list(oracle.page_codecs),
        "scope": "diagnostic allocations outside the native runtime memory budget",
        "quality_or_perplexity_measured": False,
    }
    if source_dir is not None:
        from runtime.nexapack.bundle import ModelBundleReader
        with ModelBundleReader(bundle_path) as bundle:
            provenance = bundle.manifest["provenance"]["data"]
        original_config, original_weights = load_safetensors_weights(source_dir, expected_provenance=provenance)
        if original_config.to_dict() != config.to_dict():
            raise ValueError("Reference checkpoint config does not match the bundle")
        original = llama_forward(original_config, original_weights, tokens)
        report["quantization_error"] = representation_error(quantized, original)
        report["combined_quantization_error"] = representation_error(expected, original)
    return report
