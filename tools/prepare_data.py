#!/usr/bin/env python3
"""
prepare_data.py — Prepare training data for NexaLang Transformer

Usage:
    # From a text file:
    python3 tools/prepare_data.py --input data/text/corpus_pt.txt --output data/text --split 0.9

    # Download Portuguese literature from Project Gutenberg:
    python3 tools/prepare_data.py --download pt --output data/text --split 0.9

Binary format (compatible with load_tokens in NexaLang):
    [n_tokens: i32] [vocab_size: i32] [tokens: n_tokens × u8] [vocab_table: vocab_size × u8]

Byte-level encoding (V=256): every byte is its own token, supports any language.
"""

import argparse
import os
import struct
import urllib.request
import sys

# Portuguese texts from Project Gutenberg (public domain)
# Verified to be in Portuguese language
PT_URLS = [
    # Machado de Assis - Dom Casmurro
    "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",
    # Machado de Assis - Memórias Póstumas de Brás Cubas
    "https://www.gutenberg.org/cache/epub/54829/pg54829.txt",
    # Machado de Assis - Quincas Borba
    "https://www.gutenberg.org/cache/epub/55682/pg55682.txt",
    # Machado de Assis - Esaú e Jacó
    "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",
    # Camões - Os Lusíadas
    "https://www.gutenberg.org/cache/epub/3333/pg3333.txt",
]

# Common Portuguese words for language detection
PT_WORDS = {"que", "de", "não", "para", "uma", "com", "por", "mais", "como", "seu",
            "sua", "dos", "das", "nos", "nas", "era", "foi", "tinha", "este", "esta",
            "esse", "essa", "aqui", "muito", "ainda", "também", "depois", "então",
            "quando", "onde", "nao", "tambem", "entao"}


def download_gutenberg(url: str) -> str:
    """Download a Gutenberg text, strip header/footer."""
    print(f"  Downloading {url.split('/')[-1]}...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"FAILED ({e})")
        return ""
    
    # Try UTF-8 first, fall back to latin-1
    for enc in ["utf-8", "latin-1"]:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        print("FAILED (encoding)")
        return ""
    
    # Strip Gutenberg header/footer
    start_markers = ["*** START OF THIS PROJECT", "*** START OF THE PROJECT",
                     "***START OF THIS PROJECT", "***START OF THE PROJECT"]
    end_markers = ["*** END OF THIS PROJECT", "*** END OF THE PROJECT",
                   "***END OF THIS PROJECT", "***END OF THE PROJECT"]
    
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[text.index("\n", idx) + 1:]
            break
    
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    
    text = text.strip()
    
    # Language check: verify text is Portuguese
    words = text.lower().split()
    word_set = set(words[:5000])  # Check first 5000 words
    pt_count = len(word_set & PT_WORDS)
    if pt_count < 5:
        print(f"SKIPPED (not Portuguese, only {pt_count} PT words found)")
        return ""
    
    print(f"OK ({len(text):,} chars, {pt_count} PT words)")
    return text


def prepare_byte_level(text: str, output_dir: str, split_ratio: float):
    """Encode text as byte-level tokens (V=256) and save train/val bins."""
    # Encode to UTF-8 bytes
    data = text.encode("utf-8")
    print(f"\nTotal: {len(data):,} bytes (tokens)")
    
    # Find unique bytes used
    unique_bytes = sorted(set(data))
    vocab_size = 256  # Always use full byte range
    print(f"Unique bytes used: {len(unique_bytes)}")
    print(f"Vocab size: {vocab_size}")
    
    # Byte-level: token_id = byte_value (identity mapping)
    tokens = list(data)
    
    # Split
    n = len(tokens)
    split_idx = int(n * split_ratio)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")
    
    # Vocab table: maps token_id -> byte value (identity for byte-level)
    vocab_table = bytes(range(256))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write train.bin
    train_path = os.path.join(output_dir, "train.bin")
    with open(train_path, "wb") as f:
        f.write(struct.pack("<ii", len(train_tokens), vocab_size))
        f.write(bytes(train_tokens))
        f.write(vocab_table)
    print(f"\nSaved: {train_path} ({os.path.getsize(train_path):,} bytes)")
    
    # Write val.bin
    val_path = os.path.join(output_dir, "val.bin")
    with open(val_path, "wb") as f:
        f.write(struct.pack("<ii", len(val_tokens), vocab_size))
        f.write(bytes(val_tokens))
        f.write(vocab_table)
    print(f"Saved: {val_path} ({os.path.getsize(val_path):,} bytes)")
    
    # Write vocab.bin
    vocab_path = os.path.join(output_dir, "vocab.bin")
    with open(vocab_path, "wb") as f:
        f.write(struct.pack("<i", vocab_size))
        f.write(vocab_table)
    print(f"Saved: {vocab_path}")
    
    # Show sample
    sample = text[:200]
    print(f"\nSample text:\n{'─'*60}")
    print(sample)
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for NexaLang Transformer")
    parser.add_argument("--input", type=str, help="Path to input text file")
    parser.add_argument("--download", type=str, choices=["pt", "en"], 
                        help="Download corpus: 'pt' for Portuguese literature")
    parser.add_argument("--output", type=str, default="data/text_pt",
                        help="Output directory for train.bin/val.bin")
    parser.add_argument("--split", type=float, default=0.9,
                        help="Train/val split ratio (default: 0.9)")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="Max characters to use (0 = all)")
    args = parser.parse_args()
    
    if args.input:
        print(f"Loading text from {args.input}...")
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"  Loaded {len(text):,} chars")
    elif args.download == "pt":
        print("Downloading Portuguese literature from Project Gutenberg...")
        print("(Public domain works by Machado de Assis, José de Alencar, Eça de Queirós, Camões)")
        print()
        texts = []
        for url in PT_URLS:
            t = download_gutenberg(url)
            if t:
                texts.append(t)
        text = "\n\n".join(texts)
        
        if not text:
            print("ERROR: Could not download any texts.")
            sys.exit(1)
        
        # Save raw corpus
        os.makedirs(args.output, exist_ok=True)
        corpus_path = os.path.join(args.output, "corpus_pt.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nRaw corpus saved to {corpus_path} ({len(text):,} chars)")
    else:
        parser.print_help()
        sys.exit(1)
    
    if args.max_chars > 0 and len(text) > args.max_chars:
        text = text[:args.max_chars]
        print(f"Truncated to {len(text):,} chars")
    
    prepare_byte_level(text, args.output, args.split)
    print("\nDone! Ready for training.")


if __name__ == "__main__":
    main()
