"""Q8 KV pages and the shared per-codec attention kernel they dispatch through."""
import ctypes
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.model_config import ModelConfig
from compiler.paged_kv_plan import make_paged_kv_cache_plan
from runtime.nexapack.format import decode_q4_row, decode_q8_row, quantize_q4_row, quantize_q8_row
from runtime.nexapack.transformer import _load_kernels
from test_paged_transformer_regressions import _PagedFixture

KV_CODEC_IDS = {"f32": 0, "q3": 3, "q4": 4, "q8": 8}


def tiny(**overrides):
    options = {"name": "kv_codec_fixture", "vocab_size": 12, "hidden_size": 8,
               "intermediate_size": 12, "num_hidden_layers": 1, "num_attention_heads": 4,
               "num_key_value_heads": 2, "max_position_embeddings": 12,
               "tie_word_embeddings": False, "rms_norm_eps": 1e-5, "rope_theta": 10000.0}
    options.update(overrides)
    return ModelConfig(**options)


class Q8KVPlanRegressions(unittest.TestCase):
    def test_layout_costs_a_scale_and_one_byte_per_coordinate(self):
        config = tiny()
        plan = make_paged_kv_cache_plan(config, 8, 2, codec="q8", group_size=2)
        groups = config.head_dim // 2
        self.assertEqual(plan.head_row_bytes, groups * (4 + 2))
        self.assertEqual(plan.token_bytes, config.num_key_value_heads * plan.head_row_bytes)
        described = plan.to_dict()
        self.assertEqual(described["codec_id"], "Q8_GROUPED")
        self.assertEqual(described["layout"], "token_head_grouped")
        self.assertEqual((described["group_size"], described["codec"]), (2, "q8"))

    def test_q8_sits_between_the_packed_codecs_and_f32(self):
        config = tiny()
        plans = {codec: make_paged_kv_cache_plan(config, 8, 2, codec=codec,
                                                 group_size=None if codec == "f32" else 2)
                 for codec in ("q3", "q4", "q8", "f32")}
        self.assertLessEqual(plans["q3"].token_bytes, plans["q4"].token_bytes)
        self.assertLess(plans["q4"].token_bytes, plans["q8"].token_bytes)
        self.assertLess(plans["q8"].token_bytes, plans["f32"].token_bytes)

    def test_unsupported_codecs_and_missing_group_sizes_are_rejected(self):
        config = tiny()
        for codec in ("q2", "f16", "int8"):
            with self.subTest(codec=codec), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(config, 8, 2, codec=codec, group_size=2)
        # Omitting the group size falls back to the shared default, like q4.
        self.assertEqual(make_paged_kv_cache_plan(config, 8, 2, codec="q8", group_size=None).group_size,
                         make_paged_kv_cache_plan(config, 8, 2, codec="q4", group_size=None).group_size)
        for group_size in (0, -1, 1.5):
            with self.subTest(group_size=group_size), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(config, 8, 2, codec="q8", group_size=group_size)


class SharedCodecKernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shutil
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")
        cls.kernels = _load_kernels()

    def pages(self, rows, codec, group_size, head_dim, kv_heads, page_tokens):
        """Pack token/head rows into page buffers, as the session stores them."""
        encode = {"q4": quantize_q4_row, "q8": quantize_q8_row}[codec]
        row_bytes = len(encode([0.0] * head_dim, group_size))
        pages, buffers = [], []
        for start in range(0, len(rows), page_tokens * kv_heads):
            chunk = rows[start:start + page_tokens * kv_heads]
            payload = b"".join(encode(row, group_size) for row in chunk)
            payload += bytes(page_tokens * kv_heads * row_bytes - len(payload))
            buffer = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
            buffers.append(buffer)
            pages.append(ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)))
        table = (ctypes.POINTER(ctypes.c_uint8) * len(pages))(*pages)
        return table, buffers, row_bytes

    def reference(self, query, keys, values, codec, group_size, head_dim, kv_heads, heads, tokens):
        decode = {"q4": decode_q4_row, "q8": decode_q8_row}[codec]
        encode = {"q4": quantize_q4_row, "q8": quantize_q8_row}[codec]
        stored = lambda rows: [decode(encode(row, group_size), head_dim, group_size) for row in rows]
        key_rows, value_rows = stored(keys), stored(values)
        scale = 1.0 / math.sqrt(head_dim)
        output = []
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            q = query[head * head_dim:(head + 1) * head_dim]
            scores = [sum(a * b for a, b in zip(q, key_rows[token * kv_heads + kv_head])) * scale
                      for token in range(tokens)]
            top = max(scores)
            weights = [math.exp(score - top) for score in scores]
            total = sum(weights)
            output.extend(sum(weight * value_rows[token * kv_heads + kv_head][lane]
                              for token, weight in enumerate(weights)) / total
                          for lane in range(head_dim))
        return output

    def run_codec(self, codec, group_size=2, head_dim=4, kv_heads=2, heads=4, tokens=4, page_tokens=2):
        rng = [0.3, -0.7, 1.1, 0.05, -0.4, 0.9, -1.3, 0.25]
        keys = [[rng[(index * 3 + lane) % len(rng)] for lane in range(head_dim)]
                for index in range(tokens * kv_heads)]
        values = [[rng[(index * 5 + lane + 1) % len(rng)] for lane in range(head_dim)]
                  for index in range(tokens * kv_heads)]
        query = [rng[(lane * 2) % len(rng)] for lane in range(heads * head_dim)]
        key_table, key_buffers, row_bytes = self.pages(keys, codec, group_size, head_dim, kv_heads, page_tokens)
        value_table, value_buffers, _ = self.pages(values, codec, group_size, head_dim, kv_heads, page_tokens)
        source = (ctypes.c_float * len(query))(*query)
        scratch = (ctypes.c_float * tokens)()
        output = (ctypes.c_float * len(query))()
        status = self.kernels.nexa_causal_gqa_attention_paged_codec(
            source, len(source), key_table, len(key_table), value_table, len(value_table),
            KV_CODEC_IDS[codec], page_tokens, page_tokens * kv_heads * row_bytes, group_size,
            tokens - 1, 1, heads, kv_heads, head_dim, scratch, len(scratch), output, len(output))
        expected = self.reference(query, keys, values, codec, group_size,
                                  head_dim, kv_heads, heads, tokens)
        return status, list(output), expected, (key_table, value_table, key_buffers, value_buffers, row_bytes)

    def test_the_shared_kernel_matches_the_reference_for_q8(self):
        status, output, expected, _ = self.run_codec("q8")
        self.assertEqual(status, 0)
        for produced, wanted in zip(output, expected):
            self.assertAlmostEqual(produced, wanted, places=5)

    def test_the_shared_kernel_agrees_with_the_dedicated_q4_kernel(self):
        status, shared, expected, state = self.run_codec("q4")
        self.assertEqual(status, 0)
        for produced, wanted in zip(shared, expected):
            self.assertAlmostEqual(produced, wanted, places=5)
        key_table, value_table, _, _, row_bytes = state
        # The dedicated kernel, same inputs: the dispatch path must not change
        # a result that already had its own kernel.
        rng = [0.3, -0.7, 1.1, 0.05, -0.4, 0.9, -1.3, 0.25]
        query = [rng[(lane * 2) % len(rng)] for lane in range(16)]
        source = (ctypes.c_float * 16)(*query)
        scratch = (ctypes.c_float * 4)()
        output = (ctypes.c_float * 16)()
        status = self.kernels.nexa_causal_gqa_attention_paged_q4(
            source, len(source), key_table, len(key_table), value_table, len(value_table),
            2, 2 * 2 * row_bytes, 2, 3, 1, 4, 2, 4, scratch, len(scratch), output, len(output))
        self.assertEqual(status, 0)
        for dedicated, generic in zip(output, shared):
            self.assertAlmostEqual(dedicated, generic, places=6)

    def test_the_shared_kernel_rejects_unknown_codecs_and_small_buffers(self):
        _, _, _, state = self.run_codec("q8")
        key_table, value_table, _, _, row_bytes = state
        source = (ctypes.c_float * 16)()
        scratch = (ctypes.c_float * 4)()
        output = (ctypes.c_float * 16)()
        for codec_id, scratch_count in ((7, 4), (8, 1)):
            with self.subTest(codec_id=codec_id, scratch=scratch_count):
                status = self.kernels.nexa_causal_gqa_attention_paged_codec(
                    source, len(source), key_table, len(key_table), value_table, len(value_table),
                    codec_id, 2, 2 * 2 * row_bytes, 2, 3, 1, 4, 2, 4,
                    scratch, scratch_count, output, len(output))
                self.assertNotEqual(status, 0)


class Q8KVExecutionRegressions(_PagedFixture):
    def session(self, path=None, **kwargs):
        from runtime.nexapack.paged import PagedTransformerSession
        return PagedTransformerSession(path or self.path,
                                       memory_budget=kwargs.pop("memory_budget", "1MiB"),
                                       max_sequence_length=kwargs.pop("max_sequence_length", 8),
                                       page_tokens=kwargs.pop("page_tokens", 2), **kwargs)

    def test_q8_kv_costs_more_bytes_and_less_error_than_q4(self):
        results = {}
        for codec, options in (("f32", {}), ("q4", {"kv_group_size": 4}),
                               ("q3", {"kv_group_size": 4}), ("q8", {"kv_group_size": 4})):
            with self.subTest(codec=codec), self.session(kv_codec=codec, **options) as session:
                session.prefill([1, 3, 5])
                logits = session.decode(7)
                memory = session.report()["memory"]
                results[codec] = (logits, memory["kv_encoded_bytes_per_token"])
        error = {codec: max(abs(a - b) for a, b in zip(results[codec][0], results["f32"][0]))
                 for codec in ("q3", "q4", "q8")}
        self.assertLess(error["q8"], error["q4"])
        self.assertLess(error["q4"], error["q3"])
        self.assertLess(results["q4"][1], results["q8"][1])
        self.assertLess(results["q8"][1], results["f32"][1])

    def test_chunked_prefill_and_decode_agree_with_one_shot(self):
        with self.session(kv_codec="q8", kv_group_size=4, max_chunk_length=2) as chunked, \
             self.session(kv_codec="q8", kv_group_size=4) as single:
            chunked.prefill([1, 3])
            chunked.append([5])
            single.prefill([1, 3, 5])
            # Pages are quantized per token, so chunking cannot change them.
            self.assertEqual(chunked.decode(7), single.decode(7))

    def test_the_cli_accepts_q8_and_reports_its_footprint(self):
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path),
                   "--tokens", "1,3,5", "--decode-tokens", "7", "--kv-cache", "--kv-page-tokens", "2",
                   "--kv-codec", "q8", "--kv-group-size", "4", "--max-sequence-length", "8",
                   "--tile-rows", "3", "--memory-budget", "1MiB"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["kv_codec"], "q8")
        self.assertEqual(report["token_ids"], [1, 3, 5, 7])
        memory = report["memory"]
        self.assertLess(memory["kv_encoded_bytes_per_token"], memory["kv_f32_bytes_per_token"])
        invalid = [*command[:command.index("--kv-group-size")], "--kv-group-size", "0"]
        rejected = subprocess.run(invalid, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("--kv-group-size", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
