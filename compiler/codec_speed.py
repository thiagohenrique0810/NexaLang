"""How long each weight codec takes to decode, measured on the kernel the executor runs.

The precision planner has always optimized bytes. Bytes are not time: a codec
that stores fewer bytes has to unpack them, and unpacking two-bit codes costs
more arithmetic per byte than copying floats. This module measures that cost
instead of assuming it.

What is timed is `nexa_<codec>_matmul` over one block of rows with a single
activation vector — the same call `TransformerSession` makes for every weight
matrix. It therefore includes the dequantization *and* one multiply-add per
value, because that is what the executor actually pays; a pure `decode_row`
loop would instead be dominated by one ctypes dispatch per row, a Python cost
the executor does not pay for weights.

The rate is reported per *stored byte*, so a tensor's decode time is the rate
times the bytes that tensor holds. That is the shape the planner needs: it
multiplies the same bytes it already budgets.

Limits written down on purpose. This is one host, one thread, one compiler and
one scalar kernel; the numbers move with all four. Nothing here predicts GPU
time, memory bandwidth under contention, or a model's end-to-end latency.
"""
from __future__ import annotations

import ctypes
import platform
import random
import statistics
import struct
import time

DECODE_RATE_POLICY_ID = "BLOCK_MATMUL_NS_PER_STORED_BYTE_V1"
SPEED_CODECS = ("q2", "q3", "q4", "q8", "f16", "f32")
# Bytes one value occupies once stored, per codec, excluding group scales.
_BITS = {"q2": 2, "q3": 3, "q4": 4, "q8": 8, "f16": 16, "f32": 32}
_F16 = struct.Struct("<e")
DEFAULT_COLS = 1024
DEFAULT_ROWS = 32
DEFAULT_TRIALS = 9
# A trial shorter than this is noise on a wall clock; repeats are chosen to
# reach it so the number does not depend on how fast the host happens to be.
DEFAULT_MIN_TRIAL_NS = 2_000_000
MAX_REPEATS = 1 << 16


def encode_row(codec, values, group_size):
    """One stored row, in the layout the matching kernel reads.

    The quantizers live in the pack format and are imported where they are
    used: `decode_ns` and `rate_table` are arithmetic over a report and must
    stay importable without pulling in ctypes or the native library.
    """
    if codec == "f32":
        return struct.pack(f"<{len(values)}f", *values)
    if codec == "f16":
        return b"".join(_F16.pack(value) for value in values)
    from runtime.nexapack.format import (
        quantize_q2_row, quantize_q3_row, quantize_q4_row, quantize_q8_row,
    )
    encoders = {"q2": quantize_q2_row, "q3": quantize_q3_row,
                "q4": quantize_q4_row, "q8": quantize_q8_row}
    if codec not in encoders:
        raise ValueError(f"Unsupported codec for decode-time measurement: {codec!r}")
    return encoders[codec](values, group_size)


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _caller(lib, codec, buffer, payload_bytes, left, out, rows, cols, group_size):
    """The one kernel call whose duration is the measurement."""
    if codec == "f32":
        return lambda: lib.nexa_f32_matmul(left, cols, 1, buffer, rows * cols, rows, cols, out, rows)
    if codec == "f16":
        return lambda: lib.nexa_f16_matmul(left, cols, 1, buffer, payload_bytes, rows, cols, out, rows)
    kernel = getattr(lib, f"nexa_{codec}_matmul")
    return lambda: kernel(left, cols, 1, buffer, payload_bytes, rows, cols, group_size, out, rows)


def measure_decode_rates(*, cols=DEFAULT_COLS, rows=DEFAULT_ROWS, group_size=32,
                         codecs=SPEED_CODECS, trials=DEFAULT_TRIALS,
                         min_trial_ns=DEFAULT_MIN_TRIAL_NS, seed=20260920):
    """Time each codec's matmul kernel over one block and report nanoseconds per stored byte.

    Repeats per trial are chosen from a warm-up call so every trial lasts at
    least `min_trial_ns`; the reported figure is the median trial, with the
    spread between the fastest and the slowest trial published next to it. A
    median hides a scheduler hiccup that a mean would absorb into the number
    itself, and the spread is what says whether the median can be trusted.
    """
    _positive(cols, "cols")
    _positive(rows, "rows")
    _positive(group_size, "group_size")
    _positive(trials, "trials")
    _positive(min_trial_ns, "min_trial_ns")
    if cols % group_size:
        raise ValueError("cols must be a whole number of groups")
    unknown = [codec for codec in codecs if codec not in SPEED_CODECS]
    if unknown:
        raise ValueError(f"Unsupported codecs for decode-time measurement: {', '.join(unknown)}")
    # The signatures must be the ones the executor uses; redeclaring them here
    # would let the measurement drift from the code it claims to measure.
    from runtime.nexapack.transformer import _load_kernels
    lib = _load_kernels()
    generator = random.Random(seed)
    # One fixed block of weights for every codec: the comparison is between
    # codecs, so they must encode the same numbers.
    block = [[generator.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]
    left = (ctypes.c_float * cols)(*[generator.uniform(-1.0, 1.0) for _ in range(cols)])
    out = (ctypes.c_float * rows)()
    measured = {}
    for codec in codecs:
        payload = b"".join(encode_row(codec, row, group_size) for row in block)
        size = len(payload)
        buffer = ((ctypes.c_float * (rows * cols)) if codec == "f32"
                  else (ctypes.c_uint8 * size)).from_buffer_copy(payload)
        call = _caller(lib, codec, buffer, size, left, out, rows, cols, group_size)
        start = time.perf_counter_ns()
        status = call()
        once = max(time.perf_counter_ns() - start, 1)
        if status:
            raise ValueError(f"Kernel for {codec} refused the measurement block: status {status}")
        repeats = max(1, min(MAX_REPEATS, min_trial_ns // once))
        samples = []
        for _ in range(trials):
            start = time.perf_counter_ns()
            for _ in range(repeats):
                call()
            samples.append((time.perf_counter_ns() - start) / repeats)
        median = statistics.median(samples)
        measured[codec] = {
            "kernel": f"nexa_{codec}_matmul", "payload_bytes": size, "repeats": repeats,
            "ns_per_block_median": median, "ns_per_block_min": min(samples),
            "ns_per_block_max": max(samples),
            "relative_spread": (max(samples) - min(samples)) / median if median else float("inf"),
            "ns_per_byte": median / size, "ns_per_value": median / (rows * cols),
            "stored_bits_per_value": _BITS[codec]}
    return {"policy_id": DECODE_RATE_POLICY_ID, "unit": "nanoseconds per stored byte",
            "kernel_shape": {"cols": cols, "rows": rows, "group_size": group_size,
                             "activation_vectors": 1},
            "trials": trials, "min_trial_ns": min_trial_ns, "seed": seed,
            # The number belongs to this machine. Carrying the host with it is
            # what lets a report from elsewhere be recognized as not ours.
            "host": {"platform": platform.platform(), "machine": platform.machine(),
                     "python": platform.python_version()},
            "scope": ("wall time of the block matmul the executor runs for a weight matrix: "
                      "dequantization plus one multiply-add per value, on this host, single "
                      "threaded, scalar C"),
            "not_measured": ("GPU time, bandwidth under contention and end-to-end latency; "
                             "a rate measured here does not transfer to another host"),
            "codecs": measured}


def decode_ns(rates, codec, payload_bytes):
    """Predicted decode nanoseconds for `payload_bytes` stored under `codec`.

    Integer nanoseconds, so the planner compares and sums exact numbers and two
    runs of the same report produce the same plan.
    """
    if codec not in rates:
        raise ValueError(f"No measured decode rate for codec {codec!r}")
    if type(payload_bytes) is not int or payload_bytes <= 0:
        raise ValueError(f"Decode time needs positive stored bytes for codec {codec!r}")
    return max(int(round(rates[codec] * payload_bytes)), 1)


def rate_table(report):
    """The nanoseconds-per-byte the planner uses, validated, from a calibration report."""
    block = report.get("decode_time")
    if not isinstance(block, dict):
        raise ValueError("A decode-time bound needs a calibration report with a measured "
                         "'decode_time' block; run tools/nexa_calibrate.py again")
    if block.get("policy_id") != DECODE_RATE_POLICY_ID:
        raise ValueError(f"Unsupported decode-rate policy: {block.get('policy_id')!r}")
    measured = block.get("codecs")
    if not isinstance(measured, dict) or not measured:
        raise ValueError("The 'decode_time' block measures no codec")
    rates = {}
    for codec, entry in measured.items():
        if codec not in SPEED_CODECS:
            raise ValueError(f"Unsupported measured codec in 'decode_time': {codec!r}")
        rate = entry.get("ns_per_byte") if isinstance(entry, dict) else None
        if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not rate > 0:
            raise ValueError(f"The 'decode_time' block lacks a positive ns_per_byte for {codec}")
        rates[codec] = float(rate)
    return rates
