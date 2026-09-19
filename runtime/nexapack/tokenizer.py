"""Byte-level BPE tokenizer assets: versioned, verified and byte-exact.

Every token is a byte string, so decoding a sequence this tokenizer produced
returns the original bytes exactly — there is no unknown token and no lossy
normalization. Text is split into segments before merging, and a merge never
crosses a segment boundary, which keeps whitespace, digits and punctuation
from being glued into a single token by corpus accident.

Special tokens live outside that space: they have ids of their own and are
never produced by encoding text, whatever the text contains. Writing
"<|system|>" in a prompt encodes as those literal bytes, never as the control
token, so untrusted input cannot forge a role marker.

The asset directory is content-addressed: the manifest carries the size and
SHA-256 of each file, and loading verifies them before use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import unicodedata

TOKENIZER_FORMAT = "NexaTokenizer"
TOKENIZER_VERSION = 1
TOKENIZER_MODEL = "byte_bpe"
SEGMENTATION = "class_runs_with_leading_space_and_single_digits_v1"
VOCAB_MAGIC = b"NEXATOKV"
MERGES_MAGIC = b"NEXATOKM"
MAX_VOCAB_SIZE = 1 << 20
MAX_TOKEN_BYTES = 128
MAX_ASSET_BYTES = 1 << 28
# Roles and structure the Omni protocol needs; ids stay stable across models
# of one family, so a frozen tokenizer never renumbers them.
DEFAULT_SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>",
                          "<|assistant|>", "<|tool|>", "<|tool_result|>", "<|memory|>",
                          "<|route|>", "<|json|>", "<|end|>")


def _integer(value, label, minimum=0, maximum=MAX_VOCAB_SIZE):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1 << 16)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _character_class(character):
    if character.isdigit():
        return "digit"
    if character.isalpha():
        return "letter"
    if character.isspace():
        return "space"
    return "other"


def segment(text):
    """Split text into the units BPE may merge inside, never across.

    A segment is an optional single leading space followed by a run of one
    character class. Digits are emitted one by one, so numbers never merge
    into corpus-specific chunks. Any other whitespace is its own segment,
    which keeps newlines and indentation explicit.
    """
    if not isinstance(text, str):
        raise ValueError("Tokenizer input must be str")
    segments, index, length = [], 0, len(text)
    while index < length:
        start = index
        if text[index] == " " and index + 1 < length and not text[index + 1].isspace():
            index += 1
        kind = _character_class(text[index])
        if kind == "digit":
            index += 1
        elif kind == "space":
            # Runs of the same whitespace character stay together; mixed
            # whitespace splits, so "\n\n  " is three segments.
            while index + 1 < length and text[index + 1] == text[index]:
                index += 1
            index += 1
        else:
            while index < length and _character_class(text[index]) == kind:
                index += 1
        segments.append(text[start:index])
    return segments


class NexaTokenizer:
    """Loaded tokenizer asset: encode, decode and report its own identity."""
    def __init__(self, tokens, merges, special_tokens, manifest):
        self._tokens = tuple(tokens)
        self._ids = {token: index for index, token in enumerate(self._tokens)}
        if len(self._ids) != len(self._tokens):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}
        if len(self._ranks) != len(merges):
            raise ValueError("Tokenizer merges contain a duplicate pair")
        self._special = dict(special_tokens)
        self.manifest = manifest
        self._cache = {}
        for name, identifier in self._special.items():
            if not 0 <= identifier < len(self._tokens) or self._tokens[identifier] != name.encode("utf-8"):
                raise ValueError("Special token id does not match its vocabulary entry")

    @property
    def vocab_size(self):
        return len(self._tokens)

    @property
    def special_tokens(self):
        return dict(self._special)

    def token_bytes(self, identifier):
        _integer(identifier, "token id", 0, len(self._tokens) - 1)
        return self._tokens[identifier]

    def special_id(self, name):
        if name not in self._special:
            raise ValueError(f"Unknown special token {name!r}")
        return self._special[name]

    def _merge_segment(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        parts = [bytes((value,)) for value in piece]
        while len(parts) > 1:
            best, position = None, -1
            for index in range(len(parts) - 1):
                rank = self._ranks.get((parts[index], parts[index + 1]))
                if rank is not None and (best is None or rank < best):
                    best, position = rank, index
            if position < 0:
                break
            parts[position:position + 2] = [parts[position] + parts[position + 1]]
        try:
            result = tuple(self._ids[part] for part in parts)
        except KeyError as error:
            raise ValueError("Tokenizer merge produced a token outside its vocabulary") from error
        if len(self._cache) < 1 << 16:
            self._cache[piece] = result
        return result

    def encode(self, text, *, prefix=(), suffix=()):
        """Encode text, optionally framed by explicit special token ids.

        Special markers only enter through prefix/suffix: no sequence of
        characters in `text` can produce one.
        """
        ids = [self._special_argument(item) for item in prefix]
        for piece in segment(text):
            ids.extend(self._merge_segment(piece.encode("utf-8")))
        ids.extend(self._special_argument(item) for item in suffix)
        return ids

    def _special_argument(self, item):
        identifier = self.special_id(item) if isinstance(item, str) else item
        return _integer(identifier, "special token id", 0, len(self._tokens) - 1)

    def decode_bytes(self, ids, *, skip_special=False):
        parts = []
        specials = set(self._special.values())
        for identifier in ids:
            _integer(identifier, "token id", 0, len(self._tokens) - 1)
            if skip_special and identifier in specials:
                continue
            parts.append(self._tokens[identifier])
        return b"".join(parts)

    def decode(self, ids, *, skip_special=False, errors="replace"):
        """Decode to text; byte-exact round-trip is `decode_bytes`.

        A sequence cut in the middle of a character cannot be valid UTF-8, so
        the error policy is explicit instead of silently dropping bytes.
        """
        return self.decode_bytes(ids, skip_special=skip_special).decode("utf-8", errors=errors)

    def measure(self, samples):
        """bytes/token and chars/token per domain, as the training plan requires."""
        report = {}
        for domain, text in samples.items():
            ids = self.encode(text)
            encoded = text.encode("utf-8")
            report[domain] = {"characters": len(text), "bytes": len(encoded), "tokens": len(ids),
                              "bytes_per_token": len(encoded) / len(ids) if ids else 0.0,
                              "characters_per_token": len(text) / len(ids) if ids else 0.0}
        return report

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if manifest_path.stat().st_size > MAX_ASSET_BYTES:
            raise ValueError("Tokenizer manifest exceeds the supported size")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("format") != TOKENIZER_FORMAT or manifest.get("version") != TOKENIZER_VERSION
                or manifest.get("model") != TOKENIZER_MODEL
                or manifest.get("segmentation") != SEGMENTATION):
            raise ValueError("Unsupported tokenizer format, version, model or segmentation")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {"vocab.bin", "merges.bin"}:
            raise ValueError("Tokenizer manifest must describe exactly vocab.bin and merges.bin")
        for name, entry in files.items():
            path = directory / name
            if path.resolve().parent != directory.resolve():
                raise ValueError("Tokenizer files must live in the asset directory")
            size = path.stat().st_size
            if size != entry.get("bytes") or size > MAX_ASSET_BYTES:
                raise ValueError(f"Tokenizer {name} size differs from its manifest")
            if _digest(path) != entry.get("sha256"):
                raise ValueError(f"Tokenizer {name} checksum differs from its manifest")
        tokens = _read_vocabulary(directory / "vocab.bin")
        merges = _read_merges(directory / "merges.bin", tokens)
        if len(tokens) != manifest.get("vocab_size"):
            raise ValueError("Tokenizer vocabulary size differs from its manifest")
        specials = manifest.get("special_tokens")
        if not isinstance(specials, dict):
            raise ValueError("Tokenizer manifest must list its special tokens")
        return cls(tokens, merges, specials, manifest)


def _read_vocabulary(path):
    data = path.read_bytes()
    if len(data) < 16 or data[:8] != VOCAB_MAGIC:
        raise ValueError("Invalid tokenizer vocabulary header")
    version, count = struct.unpack_from("<II", data, 8)
    if version != TOKENIZER_VERSION or not 0 < count <= MAX_VOCAB_SIZE:
        raise ValueError("Unsupported tokenizer vocabulary version or size")
    tokens, offset = [], 16
    for _ in range(count):
        if offset + 2 > len(data):
            raise ValueError("Truncated tokenizer vocabulary")
        length = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        if not 0 < length <= MAX_TOKEN_BYTES or offset + length > len(data):
            raise ValueError("Tokenizer token length is out of range")
        tokens.append(data[offset:offset + length])
        offset += length
    if offset != len(data):
        raise ValueError("Tokenizer vocabulary contains trailing bytes")
    return tokens


def _read_merges(path, tokens):
    data = path.read_bytes()
    if len(data) < 16 or data[:8] != MERGES_MAGIC:
        raise ValueError("Invalid tokenizer merges header")
    version, count = struct.unpack_from("<II", data, 8)
    if version != TOKENIZER_VERSION or count > MAX_VOCAB_SIZE:
        raise ValueError("Unsupported tokenizer merges version or size")
    if len(data) != 16 + 8 * count:
        raise ValueError("Tokenizer merges size differs from its declared count")
    merges = []
    for index in range(count):
        left, right = struct.unpack_from("<II", data, 16 + 8 * index)
        for identifier in (left, right):
            if identifier >= len(tokens):
                raise ValueError("Tokenizer merge refers to a token outside the vocabulary")
        merges.append((tokens[left], tokens[right]))
    return merges


def write_tokenizer(directory, tokens, merges, special_tokens, *, training=None):
    """Publish an asset directory; every file is written before the manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    index = {token: position for position, token in enumerate(tokens)}
    vocabulary = bytearray(VOCAB_MAGIC + struct.pack("<II", TOKENIZER_VERSION, len(tokens)))
    for token in tokens:
        if not 0 < len(token) <= MAX_TOKEN_BYTES:
            raise ValueError("Token length is out of range")
        vocabulary += struct.pack("<H", len(token)) + token
    encoded = bytearray(MERGES_MAGIC + struct.pack("<II", TOKENIZER_VERSION, len(merges)))
    for left, right in merges:
        encoded += struct.pack("<II", index[left], index[right])
    (directory / "vocab.bin").write_bytes(bytes(vocabulary))
    (directory / "merges.bin").write_bytes(bytes(encoded))
    manifest = {"format": TOKENIZER_FORMAT, "version": TOKENIZER_VERSION, "model": TOKENIZER_MODEL,
                "segmentation": SEGMENTATION, "vocab_size": len(tokens),
                "byte_fallback": True, "normalization": "none",
                "special_tokens": dict(special_tokens),
                "files": {name: {"bytes": (directory / name).stat().st_size,
                                 "sha256": _digest(directory / name)}
                          for name in ("vocab.bin", "merges.bin")},
                "training": training or {}}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def normalize_corpus_text(text):
    """NFC only, so a decomposed and a composed accent train one token."""
    return unicodedata.normalize("NFC", text)
