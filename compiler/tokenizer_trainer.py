"""Deterministic byte-level BPE training for frozen NexaTokenizer assets.

Training is an offline, reproducible step: the same corpus, vocabulary size
and special tokens always produce the same vocabulary, the same merge order
and therefore the same ids. Ties are broken by the pair's own bytes, never by
dictionary iteration order, so a rerun on another machine agrees.

The corpus is read as documents; each one is normalized to NFC and split by
the tokenizer's own segmentation, so training and encoding always see the
same units. Merges are counted over segment frequencies, not over raw text,
which keeps the pass linear in distinct segments rather than in bytes.

The result is a proposal to freeze, not a claim of quality: efficiency per
domain has to be measured before adopting a vocabulary size.
"""
from __future__ import annotations

import hashlib
from collections import Counter

from runtime.nexapack.tokenizer import (
    DEFAULT_SPECIAL_TOKENS, MAX_TOKEN_BYTES, MAX_VOCAB_SIZE,
    normalize_corpus_text, segment, write_tokenizer,
)


def _corpus_identity(documents):
    """Order-independent identity, so shard order cannot change the hash."""
    digest = hashlib.sha256()
    hashes = sorted(hashlib.sha256(document.encode("utf-8")).digest() for document in documents)
    for item in hashes:
        digest.update(item)
    return {"documents": len(documents), "corpus_sha256": digest.hexdigest(),
            "bytes": sum(len(document.encode("utf-8")) for document in documents)}


def train_byte_bpe(documents, vocab_size, *, special_tokens=DEFAULT_SPECIAL_TOKENS,
                   min_frequency=2, progress=None):
    """Return (tokens, merges, special ids, statistics) for a frozen asset."""
    if not isinstance(documents, (list, tuple)) or not documents:
        raise ValueError("Tokenizer training requires a non-empty corpus")
    specials = tuple(dict.fromkeys(special_tokens))
    if any(not isinstance(name, str) or not name for name in specials):
        raise ValueError("Special tokens must be non-empty strings")
    floor = 256 + len(specials)
    if type(vocab_size) is not int or not floor <= vocab_size <= MAX_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be an integer in [{floor}, {MAX_VOCAB_SIZE}]")
    if type(min_frequency) is not int or min_frequency < 1:
        raise ValueError("min_frequency must be a positive integer")

    counts = Counter()
    for document in documents:
        if not isinstance(document, str):
            raise ValueError("Every corpus document must be str")
        for piece in segment(normalize_corpus_text(document)):
            counts[piece.encode("utf-8")] += 1

    # Byte tokens first: every byte value exists, so nothing is ever unknown.
    tokens = [bytes((value,)) for value in range(256)]
    tokens.extend(name.encode("utf-8") for name in specials)
    if len(set(tokens)) != len(tokens):
        raise ValueError("A special token collides with another vocabulary entry")
    words = {piece: tuple(bytes((value,)) for value in piece) for piece in counts}
    merges = []
    pairs = Counter()
    for piece, parts in words.items():
        frequency = counts[piece]
        for left, right in zip(parts, parts[1:]):
            pairs[(left, right)] += frequency

    while len(tokens) < vocab_size and pairs:
        # Highest frequency wins; the pair's bytes break ties reproducibly.
        best = max(pairs.items(), key=lambda item: (item[1], item[0][0] + item[0][1]))
        pair, frequency = best
        if frequency < min_frequency or len(pair[0]) + len(pair[1]) > MAX_TOKEN_BYTES:
            del pairs[pair]
            continue
        merged = pair[0] + pair[1]
        merges.append(pair)
        tokens.append(merged)
        del pairs[pair]
        for piece, parts in list(words.items()):
            if len(parts) < 2:
                continue
            updated, index = [], 0
            while index < len(parts):
                if index + 1 < len(parts) and (parts[index], parts[index + 1]) == pair:
                    updated.append(merged)
                    index += 2
                else:
                    updated.append(parts[index])
                    index += 1
            if len(updated) == len(parts):
                continue
            frequency = counts[piece]
            for left, right in zip(parts, parts[1:]):
                pairs[(left, right)] -= frequency
                if pairs[(left, right)] <= 0:
                    del pairs[(left, right)]
            for left, right in zip(updated, updated[1:]):
                pairs[(left, right)] += frequency
            words[piece] = tuple(updated)
        if progress is not None:
            progress(len(tokens), vocab_size)

    special_ids = {name: 256 + index for index, name in enumerate(specials)}
    statistics = {"requested_vocab_size": vocab_size, "vocab_size": len(tokens),
                  "merges": len(merges), "distinct_segments": len(counts),
                  "min_frequency": min_frequency,
                  "reached_target": len(tokens) == vocab_size, **_corpus_identity(documents)}
    return tokens, merges, special_ids, statistics


def build_tokenizer(destination, documents, vocab_size, **options):
    """Train and publish an asset directory, returning its manifest."""
    tokens, merges, specials, statistics = train_byte_bpe(documents, vocab_size, **options)
    return write_tokenizer(destination, tokens, merges, specials, training=statistics)
