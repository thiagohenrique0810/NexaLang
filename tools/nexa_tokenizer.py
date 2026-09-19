#!/usr/bin/env python3
"""Train, inspect and exercise NexaTokenizer assets without PyTorch.

Training is offline and reproducible; encoding never emits a special token for
text, and decoding a sequence this tool produced returns the original bytes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.tokenizer_trainer import build_tokenizer
from runtime.nexapack.tokenizer import DEFAULT_SPECIAL_TOKENS, NexaTokenizer

MAX_DOCUMENT_BYTES = 1 << 24
MAX_DOCUMENTS = 1 << 20


def read_corpus(source):
    """Read .txt files of a directory, or one JSON Lines file with a text field."""
    path = Path(source)
    documents = []
    if path.is_dir():
        files = sorted(item for item in path.rglob("*.txt") if item.is_file())
    elif path.suffix == ".jsonl":
        files = [path]
    else:
        files = [path]
    for item in files:
        size = item.stat().st_size
        if size > MAX_DOCUMENT_BYTES:
            raise ValueError(f"{item} exceeds the {MAX_DOCUMENT_BYTES}-byte document limit")
        text = item.read_text(encoding="utf-8")
        if item.suffix == ".jsonl":
            for number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError(f"{item}:{number} must be a JSON object with a text field")
                documents.append(record["text"])
        else:
            documents.append(text)
        if len(documents) > MAX_DOCUMENTS:
            raise ValueError("Corpus exceeds the supported document count")
    if not documents:
        raise ValueError(f"No corpus documents found in {source}")
    return documents


def token_list(value):
    return [int(item) for item in value.split(",") if item != ""]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    trainer = commands.add_parser("train", help="Train and publish a tokenizer asset")
    trainer.add_argument("--corpus", required=True, help="Directory of .txt files or a .jsonl file")
    trainer.add_argument("--out", required=True, type=Path, help="Destination directory; must not exist")
    trainer.add_argument("--vocab-size", required=True, type=int)
    trainer.add_argument("--min-frequency", type=int, default=2)
    trainer.add_argument("--special-token", action="append", dest="special_tokens",
                         help="Replace the default special token list; repeat in id order")

    inspector = commands.add_parser("inspect", help="Verify an asset and report its identity")
    inspector.add_argument("tokenizer", type=Path)
    inspector.add_argument("--samples", type=Path,
                           help="JSON object mapping domain to sample text for efficiency metrics")

    encoder = commands.add_parser("encode", help="Encode text and verify the byte round-trip")
    encoder.add_argument("tokenizer", type=Path)
    encoder.add_argument("--text")
    encoder.add_argument("--input", type=Path)
    encoder.add_argument("--prefix", help="Comma-separated special token names")
    encoder.add_argument("--suffix", help="Comma-separated special token names")

    decoder = commands.add_parser("decode", help="Decode token ids back to text")
    decoder.add_argument("tokenizer", type=Path)
    decoder.add_argument("--ids", required=True, type=token_list)
    decoder.add_argument("--skip-special", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "train":
        specials = tuple(args.special_tokens) if args.special_tokens else DEFAULT_SPECIAL_TOKENS
        manifest = build_tokenizer(args.out, read_corpus(args.corpus), args.vocab_size,
                                   special_tokens=specials, min_frequency=args.min_frequency)
        report = {"tokenizer": str(args.out), **manifest}
    elif args.command == "inspect":
        tokenizer = NexaTokenizer.load(args.tokenizer)
        report = {"tokenizer": str(args.tokenizer), "verified": True, **tokenizer.manifest}
        if args.samples is not None:
            samples = json.loads(args.samples.read_text(encoding="utf-8"))
            if not isinstance(samples, dict) or not all(isinstance(item, str) for item in samples.values()):
                raise ValueError("Samples must be a JSON object of domain to text")
            report["efficiency"] = tokenizer.measure(samples)
    elif args.command == "encode":
        if (args.text is None) == (args.input is None):
            parser.error("encode requires exactly one of --text or --input")
        text = args.text if args.text is not None else args.input.read_text(encoding="utf-8")
        tokenizer = NexaTokenizer.load(args.tokenizer)
        names = {"prefix": args.prefix, "suffix": args.suffix}
        frames = {key: tuple(value.split(",")) if value else () for key, value in names.items()}
        ids = tokenizer.encode(text, **frames)
        restored = tokenizer.decode_bytes(ids, skip_special=True)
        report = {"tokenizer": str(args.tokenizer), "token_ids": ids, "token_count": len(ids),
                  "text_bytes": len(text.encode("utf-8")), "round_trip_exact": restored == text.encode("utf-8"),
                  "special_tokens_from_text": 0, "frames": {key: list(value) for key, value in frames.items()}}
        framed = set(tokenizer.special_tokens.values())
        body = ids[len(frames["prefix"]):len(ids) - len(frames["suffix"])] if ids else []
        report["special_tokens_from_text"] = sum(identifier in framed for identifier in body)
        if not report["round_trip_exact"] or report["special_tokens_from_text"]:
            raise ValueError("Encoding broke the byte round-trip or emitted a special token for text")
    else:
        tokenizer = NexaTokenizer.load(args.tokenizer)
        report = {"tokenizer": str(args.tokenizer), "token_ids": args.ids,
                  "text": tokenizer.decode(args.ids, skip_special=args.skip_special)}
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Tokenizer command failed: {error}", file=sys.stderr)
        raise SystemExit(1)
