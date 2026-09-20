#!/usr/bin/env python3
"""Execute a local Llama Q4 bundle on token IDs using native CPU kernels.

By default decode recomputes the prefix; --kv-cache enables incremental F32 KV.
Tokenization is not implemented. Optional PyTorch verification is outside the CPU budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.nexapack.transformer import TransformerSession
from tools.nexa_bench import write_report

# Page size when the caller does not choose one: small enough for short
# prompts, large enough to keep the page table modest on long contexts.
DEFAULT_PAGE_TOKENS = 16


def token_list(text):
    try:
        parts = text.split(",")
        if not parts or any(not part.strip() for part in parts):
            raise ValueError
        return [int(part.strip()) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer token IDs, e.g. 1,3,5") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--tokens", type=token_list, help="Prompt as explicit IDs; use --prompt for text")
    parser.add_argument("--prompt", help="Prompt as text, encoded by --tokenizer")
    parser.add_argument("--tokenizer", type=Path, help="NexaTokenizer asset directory")
    parser.add_argument("--bos", action="store_true", help="Frame the encoded prompt with the tokenizer's <|bos|>")
    decode = parser.add_mutually_exclusive_group()
    decode.add_argument("--decode-tokens", type=token_list, help="Append these known IDs one by one")
    decode.add_argument("--generate", type=int, default=0, help="Generate this many IDs using deterministic argmax")
    parser.add_argument("--eos-token", type=int, help="Stop greedy generation when this ID is selected")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--memory-budget", default="512MiB")
    parser.add_argument("--reserve", default="0B")
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--kv-cache", action="store_true",
                        help="Accepted for compatibility; paged KV is the default execution path")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute the whole prefix each step instead of keeping a KV cache")
    parser.add_argument("--kv-two-banks", action="store_true",
                        help="Use the two-bank reference cache of M4.00; it stores the cache twice")
    parser.add_argument("--prefill-chunk-size", type=int, help="Bound activation/logit buffers by processing the prompt in chunks; requires --kv-cache")
    parser.add_argument("--kv-page-tokens", type=int, help="Use on-demand KV pages with this token capacity per page; requires --kv-cache")
    parser.add_argument("--kv-codec", choices=("f32", "q4", "q3", "q8", "tq"), default="f32", help="KV storage codec; q4/q3/q8/tq require --kv-cache and --kv-page-tokens")
    parser.add_argument("--kv-group-size", type=int, help="Group size within each KV head for q4/q3 (default: 32)")
    parser.add_argument("--kv-bits", type=int, help="TQ bits per coordinate, 1 to 8 (default: 3)")
    parser.add_argument("--kv-seed", type=int, help="TQ signed-int32 transform seed (default: 42)")
    parser.add_argument("--kv-policy", choices=("homogeneous", "age"), default="homogeneous",
                        help="Optional CPU page aging: hot F32, warm Q4, cold Q3")
    parser.add_argument("--kv-hot-pages", type=int, help="Newest F32 pages under the age policy (default: 1)")
    parser.add_argument("--kv-warm-pages", type=int, help="Intermediate Q4 pages under the age policy (default: 1)")
    parser.add_argument("--kv-backing-store", type=Path,
                        help="Private temporary cache for cold Q3 pages; requires --kv-policy age")
    parser.add_argument("--kv-reload-slots", type=int,
                        help="Cold pages kept resident between layers and calls (default: 1); requires --kv-backing-store")
    parser.add_argument("--fork-tokens", type=token_list,
                        help="Derive a second sequence from the finished prefix and append these IDs to it; requires paged KV without a backing store")
    parser.add_argument("--verify", action="store_true", help="Compare small-model logits with optional PyTorch oracle")
    parser.add_argument("--reference-checkpoint", type=Path, help="Also measure original-vs-Q4 quantization error")
    parser.add_argument("--include-logits", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.generate < 0:
        parser.error("--generate must be nonnegative")
    if args.reference_checkpoint is not None and not args.verify:
        parser.error("--reference-checkpoint requires --verify")
    if args.prefill_chunk_size is not None and args.prefill_chunk_size <= 0:
        parser.error("--prefill-chunk-size must be a positive size")
    if args.recompute and (args.kv_cache or args.kv_two_banks or args.kv_page_tokens is not None
                           or args.kv_codec != "f32" or args.kv_policy != "homogeneous"
                           or args.kv_backing_store is not None or args.fork_tokens is not None
                           or args.prefill_chunk_size is not None):
        parser.error("--recompute keeps no cache; drop the KV options")
    if args.kv_two_banks and (args.kv_page_tokens is not None or args.kv_codec != "f32"
                              or args.kv_policy != "homogeneous" or args.kv_backing_store is not None):
        parser.error("--kv-two-banks is the F32 reference cache; it takes no paging or codec options")
    if args.kv_page_tokens is not None and args.kv_page_tokens <= 0:
        parser.error("--kv-page-tokens must be a positive size")
    if args.kv_group_size is not None and ((args.kv_policy != "age" and args.kv_codec not in ("q4", "q3", "q8"))
                                          or args.kv_group_size <= 0):
        parser.error("--kv-group-size requires --kv-codec q4/q3/q8 or --kv-policy age and a positive size")
    if args.kv_policy == "age":
        if args.recompute or args.kv_two_banks:
            parser.error("--kv-policy age needs the paged cache; drop --recompute/--kv-two-banks")
        if args.kv_codec != "f32" or args.kv_bits is not None or args.kv_seed is not None:
            parser.error("--kv-policy age fixes F32/Q4/Q3 tiers; do not pass another --kv-codec or TQ options")
        if args.kv_hot_pages is not None and args.kv_hot_pages < 1:
            parser.error("--kv-hot-pages must be positive")
        if args.kv_warm_pages is not None and args.kv_warm_pages < 0:
            parser.error("--kv-warm-pages must be nonnegative")
    elif args.kv_hot_pages is not None or args.kv_warm_pages is not None:
        parser.error("--kv-hot-pages/--kv-warm-pages require --kv-policy age")
    if args.kv_backing_store is not None and args.kv_policy != "age":
        parser.error("--kv-backing-store requires --kv-policy age")
    if args.kv_reload_slots is not None and (args.kv_backing_store is None or args.kv_reload_slots < 1):
        parser.error("--kv-reload-slots requires --kv-backing-store and at least one slot")
    if (args.tokens is None) == (args.prompt is None):
        parser.error("pass exactly one of --tokens or --prompt")
    if (args.prompt is not None) != (args.tokenizer is not None):
        parser.error("--prompt requires --tokenizer, and --tokenizer is only used with --prompt")
    if args.bos and args.prompt is None:
        parser.error("--bos requires --prompt")
    if args.fork_tokens is not None and (args.recompute or args.kv_two_banks):
        parser.error("--fork-tokens needs the paged cache; drop --recompute/--kv-two-banks")
    if (args.kv_bits is not None or args.kv_seed is not None) and args.kv_codec != "tq":
        parser.error("--kv-bits/--kv-seed require --kv-codec tq")
    if args.kv_bits is not None and not 1 <= args.kv_bits <= 8:
        parser.error("--kv-bits must be in [1, 8]")
    if args.kv_seed is not None and not -(1 << 31) <= args.kv_seed < (1 << 31):
        parser.error("--kv-seed must fit signed int32")
    try:
        for source in (args.bundle, args.reference_checkpoint):
            if (source is not None and args.report is not None
                    and args.report.resolve().is_relative_to(source.resolve())):
                raise ValueError("Reports must be written outside model and checkpoint directories")
        tokenizer = None
        if args.prompt is not None:
            from runtime.nexapack.tokenizer import NexaTokenizer
            tokenizer = NexaTokenizer.load(args.tokenizer)
            frame = ("<|bos|>",) if args.bos else ()
            args.tokens = tokenizer.encode(args.prompt, prefix=frame)
            if not args.tokens:
                raise ValueError("The encoded prompt is empty")
        steps = len(args.decode_tokens or ()) or args.generate
        forked = len(args.fork_tokens or ())
        capacity = (args.max_sequence_length if args.max_sequence_length is not None
                    else len(args.tokens) + steps + forked)
        session_type = TransformerSession
        session_options = {}
        if args.recompute:
            pass  # The baseline stays reachable, but is no longer the default.
        elif args.kv_two_banks:
            from runtime.nexapack.incremental import IncrementalTransformerSession
            session_type = IncrementalTransformerSession
        else:
            # Paged KV is the execution path now. The two-bank cache of M4.00
            # stored the whole cache twice and must be asked for by name.
            from runtime.nexapack.paged import PagedTransformerSession
            session_type = PagedTransformerSession
            session_options["page_tokens"] = (DEFAULT_PAGE_TOKENS if args.kv_page_tokens is None
                                              else args.kv_page_tokens)
            session_options["kv_group_size"] = args.kv_group_size
            if True:
                if args.kv_policy == "age":
                    from runtime.nexapack.tiered import TieredTransformerSession
                    session_type = TieredTransformerSession
                    if args.kv_backing_store is not None:
                        from runtime.nexapack.offloaded import OffloadedTieredTransformerSession
                        session_type = OffloadedTieredTransformerSession
                        session_options["kv_backing_store"] = args.kv_backing_store
                        if args.kv_reload_slots is not None:
                            session_options["kv_reload_slots"] = args.kv_reload_slots
                    session_options.update({"hot_pages": 1 if args.kv_hot_pages is None else args.kv_hot_pages,
                                            "warm_pages": 1 if args.kv_warm_pages is None else args.kv_warm_pages,
                                            "kv_group_size": 32 if args.kv_group_size is None else args.kv_group_size})
                else:
                    session_options["kv_codec"] = args.kv_codec
                    if args.kv_codec == "tq":
                        session_options.update({"kv_bits": args.kv_bits, "kv_seed": args.kv_seed})
            if args.prefill_chunk_size is not None:
                session_options["max_chunk_length"] = min(args.prefill_chunk_size, capacity)
        with session_type(args.bundle, memory_budget=args.memory_budget,
                                max_sequence_length=capacity, tile_rows=args.tile_rows,
                                reserve_bytes=args.reserve, **session_options) as session:
            if tokenizer is not None and tokenizer.vocab_size != session.config.vocab_size:
                # A prompt encoded by another vocabulary would silently index
                # the wrong embeddings instead of failing.
                raise ValueError("Tokenizer vocabulary differs from the model's vocab_size")
            if len(args.tokens) + steps > session.max_sequence_length:
                raise ValueError("Requested prompt and decode exceed the sequence capacity")
            for token in (args.decode_tokens or ()):
                if not 0 <= token < session.config.vocab_size:
                    raise ValueError("Decode token is outside the model vocabulary")
            if args.eos_token is not None and not 0 <= args.eos_token < session.config.vocab_size:
                raise ValueError("EOS token is outside the model vocabulary")
            chunk_size = args.prefill_chunk_size or len(args.tokens)
            token_chunks = [args.tokens[:chunk_size]]
            logits = session.prefill(args.tokens[:chunk_size])
            first = session.report()
            step_reports = [{"mode": "prefill", "sequence_length": len(session.token_ids),
                             "processed_tokens": first["processed_tokens"],
                             "timing": first["timing"], "io": first["io"],
                             "managed_buffers_peak_bound_bytes": first["memory"]["managed_buffers_peak_bound_bytes"]}]
            for start in range(chunk_size, len(args.tokens), chunk_size):
                token_chunks.append(args.tokens[start:start + chunk_size])
                logits.extend(session.append(token_chunks[-1]))
                current = session.report()
                step_reports.append({"mode": "prefill_append", "sequence_length": len(session.token_ids),
                                     "processed_tokens": current["processed_tokens"],
                                     "timing": current["timing"], "io": current["io"],
                                     "managed_buffers_peak_bound_bytes": current["memory"]["managed_buffers_peak_bound_bytes"]})
            generated = []
            for index in range(steps):
                token = args.decode_tokens[index] if args.decode_tokens else max(
                    range(session.config.vocab_size), key=lambda i: logits[-1][i])
                logits.append(session.decode(token))
                token_chunks.append([token])
                generated.append(token)
                current = session.report()
                step_reports.append({"mode": "decode_recompute" if args.recompute else "decode_incremental",
                                     "sequence_length": len(session.token_ids),
                                     "processed_tokens": current["processed_tokens"],
                                     "timing": current["timing"], "io": current["io"],
                                     "managed_buffers_peak_bound_bytes": current["memory"]["managed_buffers_peak_bound_bytes"]})
                if args.decode_tokens is None and token == args.eos_token:
                    break
            report = session.report()
            # The session hashes its most recent output chunk. The CLI retains
            # all causal rows, so expose a comparable full-context digest in
            # both execution modes and retain the original chunk digest too.
            report["last_chunk_logits_sha256"] = report["logits_sha256"]
            all_logits = hashlib.sha256()
            for row in logits:
                for value in row:
                    all_logits.update(struct.pack("<f", value))
            report["logits_sha256"] = all_logits.hexdigest()
            report["logits_scope"] = "full_context"
            report["logits_shape"] = [len(session.token_ids), session.config.vocab_size]
            if tokenizer is not None:
                identity = tokenizer.manifest
                report["tokenizer"] = {
                    "path": str(args.tokenizer), "format": identity["format"], "version": identity["version"],
                    "vocab_size": identity["vocab_size"], "segmentation": identity["segmentation"],
                    "files": {name: entry["sha256"] for name, entry in identity["files"].items()},
                    "framed_with_bos": args.bos}
                report["prompt"] = args.prompt
                report["prompt_bytes"] = len(args.prompt.encode("utf-8"))
                report["generated_text"] = tokenizer.decode(generated, skip_special=True) if generated else ""
                report["decoded_text"] = tokenizer.decode(list(session.token_ids), skip_special=True)
            report.update({"input_token_ids": args.tokens, "appended_token_ids": generated,
                           "generation": "provided_ids" if args.decode_tokens else "greedy_argmax",
                           "tokenizer_executed": False, "steps": step_reports,
                           "next_token_id": max(range(session.config.vocab_size), key=lambda i: logits[-1][i])})
            report["run_totals"] = {
                "managed_buffers_peak_bound_bytes": max(step["managed_buffers_peak_bound_bytes"] for step in step_reports),
                "execution_wall_seconds": sum(step["timing"]["execution_wall_seconds"] for step in step_reports),
                "q4_payload_bytes_read": sum(step["io"]["q4_payload_bytes_read"] for step in step_reports),
                "raw_payload_bytes_read": sum(step["io"]["raw_payload_bytes_read"] for step in step_reports),
                "processed_tokens": sum(step["processed_tokens"] for step in step_reports),
                "scope": "native execution calls; excludes loading, planning, compilation and optional reference",
            }
            if args.kv_policy == "age":
                report["token_chunks"] = token_chunks
                report["run_totals"].update({
                    "kv_pages_reencoded": sum(step["io"]["kv_pages_reencoded"] for step in step_reports),
                    "kv_migration_source_bytes": sum(step["io"]["kv_migration_source_bytes"] for step in step_reports),
                    "kv_migration_target_bytes": sum(step["io"]["kv_migration_target_bytes"] for step in step_reports),
                    "kv_migration_wall_seconds": sum(step["timing"]["migration_wall_seconds"] for step in step_reports),
                })
            if args.kv_backing_store is not None:
                for name in ("kv_backing_bytes_read", "kv_backing_bytes_written", "kv_page_reloads",
                             "kv_pages_evicted", "kv_reload_cache_hits", "kv_reload_cache_misses",
                             "kv_reload_bytes_avoided", "kv_reload_slot_admissions", "kv_reload_slot_evictions"):
                    report["run_totals"][name] = sum(step["io"][name] for step in step_reports)
                for name in ("kv_backing_read_seconds", "kv_backing_write_seconds"):
                    report["run_totals"][name] = sum(step["timing"][name] for step in step_reports)
            if args.fork_tokens:
                # A derived sequence continues this prefix without copying its
                # complete pages; both sequences stay independent afterwards.
                derived = session.fork()
                try:
                    adoption = derived.report()["kv_prefix_adoption"]
                    appended = derived.append(args.fork_tokens)
                    memory = derived.report()["memory"]
                    digest = hashlib.sha256()
                    for row in appended:
                        for value in row:
                            digest.update(struct.pack("<f", value))
                    report["derived_sequence"] = {
                        "prefix_adoption": adoption, "token_ids": list(derived.token_ids),
                        "appended_token_ids": list(args.fork_tokens),
                        "logits_sha256": digest.hexdigest(), "logits_scope": "appended_chunk",
                        "kv_shared_page_count": memory["kv_shared_page_count"],
                        "kv_shared_allocation_bytes": memory["kv_shared_allocation_bytes"],
                        "kv_owned_allocation_bytes": memory["kv_owned_allocation_bytes"],
                        "managed_buffers_peak_bound_bytes": memory["managed_buffers_peak_bound_bytes"],
                        "scope": "second sequence sharing the complete pages of this prefix",
                    }
                finally:
                    derived.close()
            if args.verify:
                sys.path.insert(0, str(ROOT / "tests"))
                if args.kv_policy == "age":
                    from tiered_reference import verify_tiered_bundle_forward
                    report["validation"] = verify_tiered_bundle_forward(
                        args.bundle, token_chunks, logits, page_tokens=session.page_tokens,
                        hot_pages=session.policy.hot_pages, warm_pages=session.policy.warm_pages,
                        group_size=session.policy.group_size, source_dir=args.reference_checkpoint)
                else:
                    from transformer_reference import verify_bundle_forward
                    tq_options = ({"kv_bits": session.kv_bits, "kv_seed": session.kv_seed,
                                   "kv_codebook_f32le": session.kv_codebook_f32le}
                                  if args.kv_codec == "tq" else {})
                    report["validation"] = verify_bundle_forward(
                        args.bundle, list(session.token_ids), logits, source_dir=args.reference_checkpoint,
                        kv_group_size=session.kv_group_size if args.kv_codec != "f32" else None,
                        kv_codec=args.kv_codec, **tq_options)
                report["validation"]["model_quality_measured"] = False
                report["validation"]["memory_scope"] = "reference weights and PyTorch allocations excluded from runtime budget"
            if args.include_logits:
                report["logits"] = logits
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report:
            write_report(args.report, rendered)
        print(rendered, end="")
        return 0 if not args.verify or report["validation"]["verified"] else 1
    except (OSError, ValueError, ArithmeticError, RuntimeError, MemoryError,
            ImportError, subprocess.CalledProcessError) as exc:
        print(f"Nexa model execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
