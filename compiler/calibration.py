"""Per-tensor calibration: what a codec costs in bytes, error and in logits.

Two measurements, deliberately kept apart. The static pass reads a tensor and
reports its distribution, its outliers and the error a codec introduces —
cheap, exact, and independent of any execution. The sensitivity pass executes
the model twice, once with every weight dense and once with a single tensor
packed, and reports how far the logits actually moved.

Neither is perplexity. A static error says nothing about how a layer's error
propagates, and a logit delta on untrained weights says nothing about answer
quality. Both are proxies with their limits written down, and they only become
a quality claim once a trained checkpoint exists (LLM.04b).

Sensitivity costs one execution per tensor. Variants reuse the dense and packed
payloads already written instead of re-encoding the whole model each time.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import struct

from runtime.nexapack.bundle import DENSE_CODECS, PACKED_CODECS
from runtime.nexapack.format import (
    decode_q2_row, decode_q3_row, decode_q4_row, decode_q8_row,
    quantize_q2_row, quantize_q3_row, quantize_q4_row, quantize_q8_row,
)

_F16 = struct.Struct("<e")


def _f16_round_trip(values, _group_size):
    """Half precision keeps no scale: the stored width is the whole codec."""
    try:
        return [_F16.unpack(_F16.pack(value))[0] for value in values]
    except (OverflowError, struct.error) as error:
        raise ValueError("Tensor value does not fit float16") from error

MAX_CALIBRATION_TENSORS = 4096
# Round-trip helpers per packed weight codec; dense needs none by definition.
_CODEC_ROUND_TRIP = {
    "q2": (quantize_q2_row, decode_q2_row),
    "q3": (quantize_q3_row, decode_q3_row),
    "q4": (quantize_q4_row, decode_q4_row),
    "q8": (quantize_q8_row, decode_q8_row),
}


def _finite(values, label):
    # Rows arrive as generators over a shard; materialize once, then check.
    result = list(values)
    for value in result:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label} must contain finite numbers")
    return result


def row_statistics(rows):
    """Distribution and outlier shape of a matrix, in one pass over its rows."""
    count = 0
    total = squared = 0.0
    minimum = math.inf
    maximum = -math.inf
    magnitudes = []
    for row in rows:
        values = _finite(row, "Tensor row")
        if not values:
            raise ValueError("Tensor rows must not be empty")
        for value in values:
            count += 1
            total += value
            squared += value * value
            minimum = min(minimum, value)
            maximum = max(maximum, value)
        magnitudes.append(max(abs(value) for value in values))
    if not count:
        raise ValueError("Tensor must have at least one row")
    mean = total / count
    variance = max(squared / count - mean * mean, 0.0)
    magnitudes.sort()
    largest = magnitudes[-1]
    median = magnitudes[len(magnitudes) // 2]
    return {"values": count, "rows": len(magnitudes), "min": minimum, "max": maximum,
            "mean": mean, "rms": math.sqrt(squared / count), "std": math.sqrt(variance),
            "max_abs": largest, "median_row_max_abs": median,
            # How far the worst row sticks out of the typical one: a large
            # ratio is what makes a tensor hard to quantize with one scale.
            "outlier_ratio": largest / median if median else math.inf}


def quantization_error(rows, group_size, codec="q4"):
    """Round-trip error of one packed codec, row by row, without the matrix."""
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if codec == "f16":
        quantize, decode = None, None
    elif codec not in _CODEC_ROUND_TRIP:
        raise ValueError(f"Unsupported codec for calibration: {codec!r}")
    else:
        quantize, decode = _CODEC_ROUND_TRIP[codec]
    count = 0
    squared = reference_squared = 0.0
    worst = 0.0
    for row in rows:
        values = _finite(row, "Tensor row")
        decoded = (_f16_round_trip(values, group_size) if quantize is None
                   else decode(quantize(values, group_size), len(values), group_size))
        for original, restored in zip(values, decoded):
            delta = original - restored
            squared += delta * delta
            reference_squared += original * original
            worst = max(worst, abs(delta))
            count += 1
    if not count:
        raise ValueError("Tensor must have at least one value")
    rmse = math.sqrt(squared / count)
    reference_rms = math.sqrt(reference_squared / count)
    stored = PACKED_CODECS.get(codec) or DENSE_CODECS[codec]
    return {"codec": stored, "group_size": group_size if codec != "f16" else None, "values": count,
            "max_abs_error": worst, "rmse": rmse, "reference_rms": reference_rms,
            "relative_rmse": rmse / reference_rms if reference_rms else math.inf,
            # Signal-to-noise in dB; higher is a tensor the codec handles well.
            "snr_db": 20 * math.log10(reference_rms / rmse) if rmse and reference_rms else math.inf}


def logit_delta(reference, measured):
    """Distance between two logit matrices, in absolute and relative terms."""
    if len(reference) != len(measured) or any(len(a) != len(b) for a, b in zip(reference, measured)):
        raise ValueError("Logit matrices must have the same shape")
    count = 0
    squared = reference_squared = 0.0
    worst = 0.0
    for expected, actual in zip(reference, measured):
        for left, right in zip(expected, actual):
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("Logits must be finite")
            delta = left - right
            squared += delta * delta
            reference_squared += left * left
            worst = max(worst, abs(delta))
            count += 1
    if not count:
        raise ValueError("Logit matrices must not be empty")
    rmse = math.sqrt(squared / count)
    reference_rms = math.sqrt(reference_squared / count)
    return {"positions": len(reference), "values": count, "max_abs_delta": worst, "rmse": rmse,
            "reference_rms": reference_rms,
            "relative_rmse": rmse / reference_rms if reference_rms else math.inf}


def build_variant(dense_dir, packed_dir, tensor, destination):
    """A bundle identical to the dense one, with a single tensor taken from the packed one.

    Payload files are hard-linked when the filesystem allows it, so measuring
    every tensor of a large model does not rewrite the model once per tensor.
    """
    dense_dir, packed_dir, destination = Path(dense_dir), Path(packed_dir), Path(destination)
    dense = json.loads((dense_dir / "manifest.json").read_text(encoding="utf-8"))
    packed = json.loads((packed_dir / "manifest.json").read_text(encoding="utf-8"))
    if dense["config"] != packed["config"]:
        raise ValueError("Variant bundles must share one model configuration")
    if tensor not in dense["tensors"] or tensor not in packed["tensors"]:
        raise ValueError(f"Unknown tensor for calibration: {tensor}")
    if dense["tensors"][tensor]["codec"] != "RAW_F32_MATRIX":
        raise ValueError(f"The reference tensor must be dense F32: {tensor}")
    measured = packed["tensors"][tensor]["codec"]
    if measured == dense["tensors"][tensor]["codec"]:
        raise ValueError(f"The measured tensor must differ from the reference: {tensor}")
    if measured not in set(PACKED_CODECS.values()) | set(DENSE_CODECS.values()):
        raise ValueError(f"The measured tensor must use a supported codec: {tensor}")
    if destination.exists():
        raise FileExistsError(f"Variant destination already exists: {destination}")
    manifest = dict(dense)
    manifest["tensors"] = dict(dense["tensors"])
    manifest["tensors"][tensor] = dict(packed["tensors"][tensor])
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        (staging / "tensors").mkdir(parents=True)
        for name, entry in manifest["tensors"].items():
            source = (packed_dir if name == tensor else dense_dir) / entry["path"]
            target = staging / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        for name, entry in manifest.get("assets", {}).items():
            target = staging / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dense_dir / entry["path"], target)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
