#!/usr/bin/env python3
"""
prepare_data_v2.py — Prepare large Portuguese corpus for NexaLang Transformer v5

Downloads ~30 major Portuguese/Brazilian novels from Project Gutenberg.
Target: 5-10M bytes of Portuguese text for conversational-quality training.

Usage:
    python3 tools/prepare_data_v2.py

Binary format: [n_tokens: i32] [vocab_size: i32] [tokens: n_tokens × u8] [vocab_table: 256 × u8]
"""

import os
import struct
import urllib.request
import sys
import time

# Large Portuguese novels from Project Gutenberg (verified IDs from catalog)
PT_BOOKS = [
    # === Machado de Assis (Brazilian master, dialogue-heavy) ===
    (54829, "Machado de Assis - Memórias Póstumas de Brás Cubas"),
    (55682, "Machado de Assis - Quincas Borba"),
    (55752, "Machado de Assis - Dom Casmurro"),
    (56737, "Machado de Assis - Esaú e Jacó"),
    (67162, "Machado de Assis - Helena"),
    (53101, "Machado de Assis - A Mão e a Luva"),
    (55797, "Machado de Assis - Memorial de Aires"),
    (57001, "Machado de Assis - Papéis Avulsos"),
    (67935, "Machado de Assis - Relíquias de Casa Velha"),
    (67780, "Machado de Assis - Iaiá Garcia"),
    (33056, "Machado de Assis - Histórias Sem Data"),

    # === Eça de Queirós (Portuguese realist, large novels) ===
    (40409, "Eça de Queirós - Os Maias"),
    (42942, "Eça de Queirós - O Primo Basílio"),
    (31971, "Eça de Queirós - O Crime do Padre Amaro"),
    (18220, "Eça de Queirós - A Cidade e as Serras"),
    (23145, "Eça de Queirós - A Ilustre Casa de Ramires"),
    (17515, "Eça de Queirós - A Relíquia"),
    (16384, "Eça de Queirós - O Mandarim"),
    (31347, "Eça de Queirós - Contos"),

    # === Camilo Castelo Branco (prolific Portuguese novelist) ===
    (16425, "Camilo Castelo Branco - Amor de Perdição"),
    (21406, "Camilo Castelo Branco - Novelas do Minho"),
    (17927, "Camilo Castelo Branco - A Queda dum Anjo"),

    # === Júlio Dinis (Portuguese realist) ===
    (16443, "Júlio Dinis - Uma Família Inglesa"),
    (16428, "Júlio Dinis - Os Fidalgos da Casa Mourisca"),

    # === José de Alencar (Brazilian romantic) ===
    (67724, "José de Alencar - O Guarani Vol. 1"),
    (67725, "José de Alencar - O Guarani Vol. 2"),
    (67740, "José de Alencar - Iracema"),

    # === Other major works ===
    (69187, "Aluísio Azevedo - O Cortiço"),
    (67535, "Lima Barreto - Triste Fim de Policarpo Quaresma"),
    (68541, "Raul Pompéia - O Ateneu"),
    (74475, "Bernardo Guimarães - A Escrava Isaura"),
    (45966, "Alexandre Herculano - Eurico, o Presbítero"),
    (24401, "Almeida Garrett - Viagens na Minha Terra"),
    (39618, "Raul Brandão - Húmus"),
    (3333,  "Camões - Os Lusíadas"),
]

# Common Portuguese words for language detection
PT_WORDS = {
    "que", "de", "não", "para", "uma", "com", "por", "mais", "como", "seu",
    "sua", "dos", "das", "nos", "nas", "era", "foi", "tinha", "este", "esta",
    "esse", "essa", "aqui", "muito", "ainda", "também", "depois", "então",
    "quando", "onde", "nao", "tambem", "entao", "elle", "ella", "disse",
    "olhos", "casa", "homem", "mulher", "tempo", "coisa", "senhor", "senhora",
}


def download_book(book_id: int, title: str) -> str:
    """Download a Gutenberg text, strip header/footer, verify Portuguese."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"  [{book_id}] {title}...", end=" ", flush=True)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"FAILED ({e})")
        return ""

    # Decode
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
    for marker in ["*** START OF THIS PROJECT", "*** START OF THE PROJECT",
                    "***START OF THIS PROJECT", "***START OF THE PROJECT"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[text.index("\n", idx) + 1:]
            break

    for marker in ["*** END OF THIS PROJECT", "*** END OF THE PROJECT",
                    "***END OF THIS PROJECT", "***END OF THE PROJECT"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    text = text.strip()

    if len(text) < 5000:
        print(f"SKIPPED (too short: {len(text)} chars)")
        return ""

    # Language check
    words = text.lower().split()
    word_set = set(words[:5000])
    pt_count = len(word_set & PT_WORDS)
    if pt_count < 5:
        print(f"SKIPPED (not Portuguese, {pt_count} PT words)")
        return ""

    print(f"OK ({len(text):,} chars, {pt_count} PT words)")
    return text


def prepare_data(texts: list[str], output_dir: str, split_ratio: float = 0.9):
    """Encode as byte-level tokens (V=256) and save train/val bins."""
    combined = "\n\n".join(texts)
    data = combined.encode("utf-8")
    print(f"\nTotal corpus: {len(data):,} bytes (tokens)")

    unique_bytes = sorted(set(data))
    print(f"Unique bytes used: {len(unique_bytes)}")
    print(f"Vocab size: 256 (byte-level)")

    tokens = list(data)
    n = len(tokens)
    split_idx = int(n * split_ratio)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")

    vocab_table = bytes(range(256))
    os.makedirs(output_dir, exist_ok=True)

    # Write train.bin
    train_path = os.path.join(output_dir, "train.bin")
    with open(train_path, "wb") as f:
        f.write(struct.pack("<ii", len(train_tokens), 256))
        f.write(bytes(train_tokens))
        f.write(vocab_table)
    print(f"\nSaved: {train_path} ({os.path.getsize(train_path):,} bytes)")

    # Write val.bin
    val_path = os.path.join(output_dir, "val.bin")
    with open(val_path, "wb") as f:
        f.write(struct.pack("<ii", len(val_tokens), 256))
        f.write(bytes(val_tokens))
        f.write(vocab_table)
    print(f"Saved: {val_path} ({os.path.getsize(val_path):,} bytes)")


def main():
    output_dir = "data/text_pt_v2"
    print("=" * 70)
    print("NexaLang Portuguese Corpus v2 — Large Novel Collection")
    print(f"Target: 5-10M tokens from {len(PT_BOOKS)} books")
    print("=" * 70)
    print()

    texts = []
    total_chars = 0
    success = 0

    for book_id, title in PT_BOOKS:
        text = download_book(book_id, title)
        if text:
            texts.append(text)
            total_chars += len(text)
            success += 1
        time.sleep(0.5)  # Be polite to Gutenberg

    print(f"\n{'=' * 70}")
    print(f"Downloaded: {success}/{len(PT_BOOKS)} books")
    print(f"Total chars: {total_chars:,}")
    print(f"{'=' * 70}")

    if not texts:
        print("ERROR: No texts downloaded!")
        sys.exit(1)

    # Save raw corpus
    os.makedirs(output_dir, exist_ok=True)
    corpus_path = os.path.join(output_dir, "corpus_pt_v2.txt")
    combined = "\n\n".join(texts)
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"\nRaw corpus: {corpus_path} ({len(combined):,} chars)")

    prepare_data(texts, output_dir)

    # Show sample
    sample = combined[:300]
    print(f"\nSample text:\n{'─' * 60}")
    print(sample)
    print(f"{'─' * 60}")
    print("\nDone! Ready for training with transformer_lm_v5.nxl")


if __name__ == "__main__":
    main()
