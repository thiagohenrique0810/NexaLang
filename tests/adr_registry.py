"""Machine-checked ADR registry: the code is the oracle, the ADRs are the claim.

Three separable pieces live here.

`walk_contract_identifiers` imports every module under the named packages and
collects the module-level names that spell a storage contract -- a magic, a
format or schema version, a codec id, a codec version, a transform id, a policy
id. It is the oracle. It **raises** when a module cannot be imported: an
extractor that skips a module in silence makes the bijection pass by finding
nothing, which is exactly the failure this file exists to prevent.

`parse_adr` reads one ADR: its front matter, the identifiers it claims, its
`adr-measurement` blocks and its prior-art references. It is strict on purpose
and raises `AdrFormatError` on anything it does not recognise, so a malformed
ADR fails loudly instead of quietly claiming nothing.

`SOURCES` holds one callable per recomputed measurement. Each one re-obtains a
number from the real implementation -- by writing a real file, by asking the
real writer for a real width -- so that a number an ADR asserts and the code
contradicts brings the suite down. `assert_source_touches_implementation`
refuses a source that never reaches `compiler` or `runtime`, because a source
that returns a literal would agree with the ADR no matter what the code did.

Prior art (P01-P31) is a bibliography carried by the planning PDFs. It is
admissible here as a statement of the problem or of a solution category, and
never as independent verification of anything this repository measures; the
parser enforces that by refusing any other role.
"""
from __future__ import annotations

import ast
import importlib
import json
import pkgutil
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADR_DIR = ROOT / "docs" / "adr"

# ---------------------------------------------------------------------------
# The contract-identifier rule.
# ---------------------------------------------------------------------------

#: Names that are a contract identifier on their own, ignoring leading
#: underscores: a private name is still a contract the format depends on.
CONTRACT_EXACT = ("MAGIC", "FORMAT", "FORMAT_VERSION", "SCHEMA_VERSION")

#: Suffixes that make a name a contract identifier.
CONTRACT_SUFFIXES = (
    "CODEC_ID",
    "CODEC_IDS",
    "CODEC_VERSION",
    "CODECS",
    "TRANSFORM_ID",
    "POLICY_ID",
    "POLICY_IDS",
)

#: Packages walked by the census.
CONTRACT_PACKAGES = ("compiler", "runtime.nexapack")


def is_contract_name(name: str) -> bool:
    """Say whether a module-level name spells a storage contract identifier."""
    if not isinstance(name, str) or not name:
        return False
    bare = name.lstrip("_")
    if not bare or bare != bare.upper():
        return False
    if bare in CONTRACT_EXACT:
        return True
    return any(bare.endswith(suffix) for suffix in CONTRACT_SUFFIXES)


_SCALARS = (str, bytes, bool, int, float)


def flatten_scalars(value) -> tuple:
    """Flatten the strings and numbers a contract value carries, in order.

    A codec id may be bound directly, or be one value in a mapping from storage
    dtype to codec id, or one element of a ladder. Flattening both keys and
    values means that adding `Q5_GROUPED` to a mapping changes the flattened
    result, so the ADR that declared the old one stops matching.
    """
    if isinstance(value, _SCALARS):
        return (value,)
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(flatten_scalars(key))
            out.extend(flatten_scalars(item))
        return tuple(out)
    if isinstance(value, (tuple, list)):
        out = []
        for item in value:
            out.extend(flatten_scalars(item))
        return tuple(out)
    if isinstance(value, (set, frozenset)):
        out = []
        for item in sorted(value, key=repr):
            out.extend(flatten_scalars(item))
        return tuple(out)
    raise TypeError(f"Unsupported contract value of type {type(value).__name__}")


def declared_text(value) -> str:
    """Render a contract value the way an ADR declares it."""
    flat = flatten_scalars(value)
    return repr(flat[0]) if len(flat) == 1 else repr(flat)


@dataclass(frozen=True)
class ContractIdentifier:
    """One contract identifier as the code currently defines it."""
    module: str
    name: str
    declared: str

    @property
    def key(self) -> str:
        return f"{self.module}:{self.name}"


class ExtractorError(RuntimeError):
    """A module under a walked package could not be imported or read."""


def walk_contract_identifiers(packages=CONTRACT_PACKAGES) -> dict:
    """Return every contract identifier the named packages define.

    Import failures are raised, never swallowed. The whole value of this
    function is that it cannot come back empty while the code still has
    contracts, and a broken import that produced a shorter census would hide
    exactly the identifiers an ADR forgot to claim.
    """
    found = {}
    for package_name in packages:
        try:
            package = importlib.import_module(package_name)
        except Exception as error:  # pragma: no cover - a broken tree
            raise ExtractorError(f"Cannot import package {package_name}: {error}") from error
        modules = {package_name: package}
        for info in pkgutil.walk_packages(package.__path__, package_name + "."):
            try:
                modules[info.name] = importlib.import_module(info.name)
            except Exception as error:
                raise ExtractorError(f"Cannot import module {info.name}: {error}") from error
        for module_name, module in modules.items():
            for attribute, value in vars(module).items():
                if not is_contract_name(attribute):
                    continue
                identifier = ContractIdentifier(module_name, attribute, declared_text(value))
                found[identifier.key] = identifier
    return found


# ---------------------------------------------------------------------------
# ADR parsing.
# ---------------------------------------------------------------------------

class AdrFormatError(ValueError):
    """An ADR does not follow the template in docs/adr/README.md."""


#: The roles prior art may take. Anything that would read as independent
#: confirmation is absent by design, and the parser refuses it.
PRIOR_ART_ROLES = ("problem", "solution-category")

#: Sections every ADR must carry, by heading text.
REQUIRED_SECTIONS = (
    "Contexto",
    "Problema técnico",
    "Decisão",
    "Medições próprias",
    "Alternativas descartadas",
    "Limites declarados",
)

_ADR_ID = re.compile(r"^ADR-\d{4}$")
_MEASUREMENT_FENCE = re.compile(r"^```adr-measurement\s*$")
_FENCE_END = re.compile(r"^```\s*$")
_HEADING = re.compile(r"^##\s+(.+?)\s*$")

#: The literal marker a measurement uses when its number can only be quoted.
CITED = "CITED"


@dataclass(frozen=True)
class ClaimedIdentifier:
    name: str
    value: str


@dataclass(frozen=True)
class Measurement:
    name: str
    value: object
    unit: str
    source: str
    cited_from: str = ""

    @property
    def is_cited(self) -> bool:
        return self.source == CITED


@dataclass(frozen=True)
class PriorArt:
    id: str
    role: str
    note: str


@dataclass
class Adr:
    path: Path
    id: str
    title: str
    status: str
    identifiers: list = field(default_factory=list)
    prior_art: list = field(default_factory=list)
    measurements: list = field(default_factory=list)
    sections: list = field(default_factory=list)


def _literal(text, where):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as error:
        raise AdrFormatError(f"{where}: {text!r} is not a Python literal") from error


def _split_front_matter(lines, path):
    if not lines or lines[0].rstrip() != "---":
        raise AdrFormatError(f"{path}: must open with a '---' front-matter fence")
    for index in range(1, len(lines)):
        if lines[index].rstrip() == "---":
            return lines[1:index], lines[index + 1:]
    raise AdrFormatError(f"{path}: front matter is never closed")


def _parse_front_matter(lines, path):
    """Parse the strict subset of YAML the template uses.

    Scalars are `key: value`. Lists are `key:` followed by `  - field: value`
    for the first field of each entry and `    field: value` for the rest.
    Anything else raises, because a line this parser silently ignored would be
    a claim nobody checks.
    """
    scalars, lists = {}, {}
    current_key, current_entry = None, None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            if line.rstrip().endswith(":") and ": " not in line:
                current_key = line.rstrip()[:-1]
                if current_key in lists or current_key in scalars:
                    raise AdrFormatError(f"{path}: duplicate front-matter key {current_key!r}")
                lists[current_key] = []
                current_entry = None
                continue
            if ": " not in line:
                raise AdrFormatError(f"{path}: cannot parse front-matter line {line!r}")
            key, value = line.split(": ", 1)
            if key in scalars or key in lists:
                raise AdrFormatError(f"{path}: duplicate front-matter key {key!r}")
            if value.strip() == "[]":
                # An ADR that claims nothing says so explicitly. Reading this as
                # a string would make an empty list look like a missing one.
                lists[key] = []
            else:
                scalars[key] = value.strip()
            current_key, current_entry = None, None
            continue
        stripped = line.strip()
        if current_key is None:
            raise AdrFormatError(f"{path}: indented line {line!r} outside a list")
        if stripped.startswith("- "):
            if ": " not in stripped[2:]:
                raise AdrFormatError(f"{path}: list entry {line!r} needs 'field: value'")
            key, value = stripped[2:].split(": ", 1)
            current_entry = {key: value.strip()}
            lists[current_key].append(current_entry)
            continue
        if current_entry is None:
            raise AdrFormatError(f"{path}: continuation line {line!r} before any '-' entry")
        if ": " not in stripped:
            raise AdrFormatError(f"{path}: cannot parse continuation line {line!r}")
        key, value = stripped.split(": ", 1)
        if key in current_entry:
            raise AdrFormatError(f"{path}: duplicate field {key!r} in a list entry")
        current_entry[key] = value.strip()
    return scalars, lists


def _parse_measurements(body, path):
    measurements, index = [], 0
    while index < len(body):
        if not _MEASUREMENT_FENCE.match(body[index].rstrip("\n")):
            index += 1
            continue
        index += 1
        fields = {}
        while index < len(body) and not _FENCE_END.match(body[index].rstrip("\n")):
            line = body[index].strip()
            index += 1
            if not line:
                continue
            if ": " not in line:
                raise AdrFormatError(f"{path}: cannot parse measurement line {line!r}")
            key, value = line.split(": ", 1)
            if key in fields:
                raise AdrFormatError(f"{path}: duplicate measurement field {key!r}")
            fields[key] = value.strip()
        if index >= len(body):
            raise AdrFormatError(f"{path}: an adr-measurement block is never closed")
        index += 1
        missing = {"name", "value", "unit", "source"} - set(fields)
        if missing:
            raise AdrFormatError(f"{path}: measurement missing {sorted(missing)}")
        unknown = set(fields) - {"name", "value", "unit", "source", "cited_from"}
        if unknown:
            raise AdrFormatError(f"{path}: measurement has unknown fields {sorted(unknown)}")
        source = fields["source"]
        cited_from = fields.get("cited_from", "")
        if source == CITED and not cited_from:
            raise AdrFormatError(f"{path}: a CITED measurement must name cited_from")
        if source != CITED and cited_from:
            raise AdrFormatError(f"{path}: a recomputed measurement must not carry cited_from")
        measurements.append(Measurement(
            name=fields["name"],
            value=_literal(fields["value"], f"{path} measurement {fields['name']}"),
            unit=fields["unit"],
            source=source,
            cited_from=cited_from,
        ))
    return measurements


def parse_adr(path) -> Adr:
    """Read one ADR file, refusing anything the template does not allow."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    front, body = _split_front_matter(lines, path)
    scalars, lists = _parse_front_matter(front, path)

    missing = {"adr", "title", "status"} - set(scalars)
    if missing:
        raise AdrFormatError(f"{path}: front matter missing {sorted(missing)}")
    if not _ADR_ID.match(scalars["adr"]):
        raise AdrFormatError(f"{path}: 'adr' must look like ADR-0001, not {scalars['adr']!r}")
    for key in ("identifiers", "prior_art"):
        if key not in lists:
            raise AdrFormatError(f"{path}: front matter must carry a '{key}:' list, even if empty")

    identifiers = []
    for entry in lists["identifiers"]:
        if set(entry) != {"name", "value"}:
            raise AdrFormatError(f"{path}: an identifier needs exactly name and value")
        identifiers.append(ClaimedIdentifier(entry["name"], entry["value"]))

    prior_art = []
    for entry in lists["prior_art"]:
        if set(entry) != {"id", "role", "note"}:
            raise AdrFormatError(f"{path}: prior art needs exactly id, role and note")
        if entry["role"] not in PRIOR_ART_ROLES:
            raise AdrFormatError(
                f"{path}: prior-art role {entry['role']!r} is not one of {list(PRIOR_ART_ROLES)}; "
                "prior art is never independent verification")
        prior_art.append(PriorArt(entry["id"], entry["role"], entry["note"]))

    sections = [match.group(1) for match in
                (_HEADING.match(line) for line in body) if match]
    return Adr(path=path, id=scalars["adr"], title=scalars["title"], status=scalars["status"],
               identifiers=identifiers, prior_art=prior_art,
               measurements=_parse_measurements(body, path), sections=sections)


def load_adrs(directory=ADR_DIR) -> list:
    """Read every ADR in the directory, ordered by id."""
    directory = Path(directory)
    if not directory.is_dir():
        raise AdrFormatError(f"No ADR directory at {directory}")
    adrs = [parse_adr(path) for path in sorted(directory.glob("ADR-*.md"))]
    if not adrs:
        raise AdrFormatError(f"No ADR files in {directory}")
    return adrs


# ---------------------------------------------------------------------------
# Measurement sources: every one re-obtains its number from the real code.
# ---------------------------------------------------------------------------

from compiler import calibration as _calibration  # noqa: E402
from compiler import codec_speed as _codec_speed  # noqa: E402
from compiler import model_plasticity as _plasticity  # noqa: E402
from compiler import precision_map as _precision_map  # noqa: E402
from compiler import tiered_kv_plan as _tiered_plan  # noqa: E402
from compiler.model_config import ModelConfig as _ModelConfig  # noqa: E402
from compiler.model_lowering import lower_model as _lower_model  # noqa: E402
from compiler.paged_kv_plan import PagedKVCachePlan as _PagedKVCachePlan  # noqa: E402
from compiler.planner import compression as _compression  # noqa: E402
from runtime.nexapack import bundle as _bundle  # noqa: E402
from runtime.nexapack import container as _container  # noqa: E402
from runtime.nexapack import format as _fmt  # noqa: E402
from runtime.nexapack.admission import SessionMemoryPool as _SessionMemoryPool  # noqa: E402
from runtime.nexapack.tq import tq_row_bytes as _tq_row_bytes  # noqa: E402

#: The fixture the mixed-codec cost table is measured on. 1024 rows of 64
#: columns at group 32: a q4 row is 40 bytes and a q2 row 24, so demoting half
#: the rows saves exactly 8192 payload bytes whatever the block count is. That
#: is what lets the block count be the only variable.
MIXED_ROWS, MIXED_COLS, MIXED_GROUP = 1024, 64, 32
MIXED_PAYLOAD_SAVED_BYTES = 8192


def _constant_rows(rows, cols):
    return ([0.5] * cols for _ in range(rows))


def _index_bytes(path):
    raw = path.read_bytes()
    _magic, _version, _flags, json_length, _offset, _total, _sha = _fmt.HEADER.unpack(raw[:64])
    return json_length


def _mixed_pair(block_count):
    """Write the same matrix uniformly and mixed; return the two paths' stats."""
    block_rows = MIXED_ROWS // block_count
    codecs = ["q2" if index < block_count // 2 else "q4" for index in range(block_count)]
    with tempfile.TemporaryDirectory() as directory:
        uniform = Path(directory) / "uniform.nxp"
        mixed = Path(directory) / "mixed.nxp"
        _fmt.write_grouped_matrix(uniform, MIXED_ROWS, MIXED_COLS, MIXED_GROUP,
                                  _constant_rows(MIXED_ROWS, MIXED_COLS),
                                  block_rows=block_rows, codec=_fmt.CODEC_ID)
        _fmt.write_mixed_matrix(mixed, MIXED_ROWS, MIXED_COLS, MIXED_GROUP,
                                _constant_rows(MIXED_ROWS, MIXED_COLS),
                                block_codecs=codecs, block_rows=block_rows)
        with _fmt.NexaPackReader(uniform) as reader_u, _fmt.NexaPackReader(mixed) as reader_m:
            payload_saved = (sum(block["size"] for block in reader_u.blocks)
                             - sum(block["size"] for block in reader_m.blocks))
        return {
            "index_delta": _index_bytes(mixed) - _index_bytes(uniform),
            "file_delta": mixed.stat().st_size - uniform.stat().st_size,
            "payload_saved": payload_saved,
        }


def _max_blocks_under_metadata_cap(mixed):
    """Largest block count whose index still fits the 1 MiB metadata cap."""
    def fits(block_count):
        block_rows = max(1, MIXED_ROWS // block_count)
        rows = block_count * block_rows
        try:
            if mixed:
                codecs = ["q2" if i < block_count // 2 else "q4" for i in range(block_count)]
                metadata, _, _ = _fmt._new_mixed_metadata(rows, MIXED_COLS, MIXED_GROUP,
                                                          block_rows, codecs)
            else:
                metadata, _, _ = _fmt._new_metadata(rows, MIXED_COLS, MIXED_GROUP,
                                                    block_rows, _fmt.CODEC_ID)
        except _fmt.NexaPackError:
            return False
        return len(_fmt._json_bytes(metadata)) <= _fmt.MAX_METADATA_BYTES

    low, high, best = 1, _fmt.MAX_BLOCKS, 0
    while low <= high:
        middle = (low + high) // 2
        if fits(middle):
            best, low = middle, middle + 1
        else:
            high = middle - 1
    return best


def _grouped_bytes_per_value(codec):
    """Ask the real writer how wide a 1024-column row is under one codec."""
    return _fmt._row_bytes(1024, 32, codec) / 1024


def _kv_bytes_per_token(codec):
    """Ask a real paged plan for K+V bytes per token on the D64/P16/G32 fixture.

    One key/value head of 64 dimensions, pages of 16 tokens, group 32: the same
    shape the KV records compare F32, Q4 and Q3 on. The number comes from the
    plan the runtime allocates against, not from arithmetic repeated here.
    """
    config = _ModelConfig(name="adr-kv", vocab_size=64, hidden_size=64,
                          intermediate_size=128, num_hidden_layers=1,
                          num_attention_heads=1, num_key_value_heads=1,
                          max_position_embeddings=64)
    plan = _PagedKVCachePlan(config=config, capacity=64, page_tokens=16, codec=codec,
                             group_size=None if codec == "f32" else 32)
    return 2 * plan.token_bytes


def _tq_codebook_entries(bits):
    """Read the codebook width the TQ index reserves, in float32 entries."""
    metadata, _offset, _total = _fmt._new_tq_metadata(4, 64, bits, 42, None, 2)
    return len(metadata["codebook_f32le"]) // 8


def _uniform_block_keys_on_disk():
    """Read the block keys a uniform file actually stores, not what a reader shows."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "uniform.nxp"
        _fmt.write_grouped_matrix(path, 4, 64, 32, _constant_rows(4, 64),
                                  block_rows=2, codec=_fmt.CODEC_ID)
        raw = path.read_bytes()
        _m, _v, _f, json_length, _o, _t, _s = _fmt.HEADER.unpack(raw[:64])
        index = json.loads(raw[64:64 + json_length])
        return tuple(sorted(index["blocks"][0]))


def _container_bytes_for_small_files(count):
    """Ask the real layout what a manifest plus `count` 20-byte tensors costs.

    Twenty bytes is two hundred times smaller than the alignment, so whatever
    comes back is the price of the 4096-byte slot, not of the data.
    """
    records = [("manifest", _container.MANIFEST_NAME, 64)]
    records += [("tensor", f"tensors/{index:04d}.bin", 20) for index in range(count)]
    _index, _payload_offset, total = _container._layout(records)
    return total


def _graph_ops(layers):
    """Lower a real model of `layers` layers and count the operations."""
    config = _ModelConfig(name="adr-probe", vocab_size=64, hidden_size=16,
                          intermediate_size=32, num_hidden_layers=layers,
                          num_attention_heads=2, num_key_value_heads=1,
                          max_position_embeddings=32)
    return len(_lower_model(config, 4).ops)


#: name -> callable. A test runs each one and compares it to what the ADR says.
SOURCES = {
    # ADR-0001, NexaPack v1 envelope.
    "nexapack_header_bytes": lambda: _fmt.HEADER.size,
    "nexapack_alignment_bytes": lambda: _fmt.ALIGNMENT,
    "nexapack_max_metadata_bytes": lambda: _fmt.MAX_METADATA_BYTES,
    "nexapack_max_blocks": lambda: _fmt.MAX_BLOCKS,
    "nexapack_index_format_name": lambda: _fmt._new_metadata(4, 64, 32, 2, _fmt.CODEC_ID)[0]["format"],
    "nexapack_header_struct": lambda: _fmt.HEADER.format,

    # ADR-0002, the grouped codec ladder.
    "q2_bytes_per_value_group32": lambda: _grouped_bytes_per_value(_fmt.Q2_CODEC_ID),
    "q3_bytes_per_value_group32": lambda: _grouped_bytes_per_value(_fmt.Q3_CODEC_ID),
    "q4_bytes_per_value_group32": lambda: _grouped_bytes_per_value(_fmt.CODEC_ID),
    "q8_bytes_per_value_group32": lambda: _grouped_bytes_per_value(_fmt.Q8_CODEC_ID),
    "q2_levels": lambda: _fmt._CODEC_LEVELS[_fmt.Q2_CODEC_ID],
    "q3_levels": lambda: _fmt._CODEC_LEVELS[_fmt.Q3_CODEC_ID],
    "q4_levels": lambda: _fmt._CODEC_LEVELS[_fmt.CODEC_ID],
    "q8_levels": lambda: _fmt._CODEC_LEVELS[_fmt.Q8_CODEC_ID],
    "grouped_scale_bytes": lambda: _fmt._group_bytes(8, _fmt.Q8_CODEC_ID) - 8,

    # ADR-0003, the dense codecs.
    "f16_bytes_per_value": lambda: float(_bundle.DENSE_WIDTH[_bundle.DENSE_CODECS["f16"]]),
    "f32_bytes_per_value": lambda: float(_bundle.DENSE_WIDTH[_bundle.DENSE_CODECS["f32"]]),
    "f16_bits": lambda: _bundle.CODEC_BITS[_bundle.DENSE_CODECS["f16"]],
    "f32_bits": lambda: _bundle.CODEC_BITS[_bundle.DENSE_CODECS["f32"]],

    # ADR-0004, the bundle manifest.
    "bundle_matrix_codec_count": lambda: len(_bundle.MATRIX_CODECS),
    "bundle_max_tensors": lambda: _bundle.MAX_TENSORS,
    "bundle_max_manifest_bytes": lambda: _bundle.MAX_MANIFEST_BYTES,
    "bundle_max_assets": lambda: _bundle.MAX_ASSETS,
    "calibration_codec_ladder": lambda: tuple(_calibration.PACKED_CODECS) + tuple(_calibration.DENSE_CODECS),

    # ADR-0005, TurboQuant storage.
    "tq_row_bytes_64_bits3": lambda: _tq_row_bytes(64, 3),
    "tq_row_bytes_64_bits4": lambda: _tq_row_bytes(64, 4),
    "tq_codebook_entries_bits3": lambda: _tq_codebook_entries(3),

    # ADR-0006, the .nxb container.
    "nxb_alignment_bytes": lambda: _container.ALIGNMENT,
    "nxb_max_sections": lambda: _container.MAX_SECTIONS,
    "nxb_section_kinds": lambda: tuple(sorted(_container.SECTION_PREFIX)),
    "nxb_bytes_for_one_small_tensor": lambda: _container_bytes_for_small_files(1),
    "nxb_bytes_for_eight_small_tensors": lambda: _container_bytes_for_small_files(8),

    # ADR-0007, mixed codecs per row block.
    "mixed_index_delta_64_blocks": lambda: _mixed_pair(64)["index_delta"],
    "mixed_file_delta_64_blocks": lambda: _mixed_pair(64)["file_delta"],
    "mixed_index_delta_256_blocks": lambda: _mixed_pair(256)["index_delta"],
    "mixed_file_delta_256_blocks": lambda: _mixed_pair(256)["file_delta"],
    "mixed_index_delta_512_blocks": lambda: _mixed_pair(512)["index_delta"],
    "mixed_file_delta_512_blocks": lambda: _mixed_pair(512)["file_delta"],
    "mixed_index_delta_1024_blocks": lambda: _mixed_pair(1024)["index_delta"],
    "mixed_file_delta_1024_blocks": lambda: _mixed_pair(1024)["file_delta"],
    "mixed_payload_saved_bytes": lambda: _mixed_pair(256)["payload_saved"],
    "uniform_max_blocks_under_metadata_cap": lambda: _max_blocks_under_metadata_cap(False),
    "mixed_max_blocks_under_metadata_cap": lambda: _max_blocks_under_metadata_cap(True),
    "uniform_block_keys_on_disk": _uniform_block_keys_on_disk,

    # ADR-0008, precision and compression policy.
    "precision_codec_ladder": lambda: tuple(_precision_map.CODECS),
    "speed_codec_ladder": lambda: tuple(_codec_speed.SPEED_CODECS),
    "precision_cost_bases": lambda: tuple(sorted(_precision_map.POLICY_IDS)),
    "precision_payload_policy_id": lambda: _precision_map.POLICY_IDS["payload"],
    "precision_physical_policy_id": lambda: _precision_map.POLICY_IDS["physical"],
    "decode_rate_policy_id": lambda: _codec_speed.DECODE_RATE_POLICY_ID,
    "compression_planner_policy_id": lambda: _compression.PLANNER_POLICY_ID,

    # ADR-0009, KV plans and tiers.
    "kv_bytes_per_token_f32_d64": lambda: _kv_bytes_per_token("f32"),
    "kv_bytes_per_token_q4_d64": lambda: _kv_bytes_per_token("q4"),
    "kv_bytes_per_token_q3_d64": lambda: _kv_bytes_per_token("q3"),
    "tier_codec_ladder": lambda: tuple(_tiered_plan._CODECS[tier]
                                       for tier in ("hot", "warm", "cold")),

    # ADR-0010, the ModelIR graph.
    "graph_ops_one_layer": lambda: _graph_ops(1),
    "graph_ops_sixteen_layers": lambda: _graph_ops(16),
    "graph_ops_thirty_two_layers": lambda: _graph_ops(32),

    # ADR-0011, plasticity.
    "plasticity_region_class_count": lambda: len(_plasticity.REGION_CLASSES),
    "plasticity_region_classes": lambda: tuple(_plasticity.REGION_CLASSES),

    # ADR-0012, joint admission.
    "admission_policy_id": lambda: _SessionMemoryPool(1024).to_dict()["policy_id"],
    "admission_schema_version": lambda: _SessionMemoryPool(1024).to_dict()["schema_version"],
    "admission_refuses_over_limit": lambda: _admission_refusal(),
}


def _admission_refusal():
    """Admit up to a shared ceiling and report how many of ten bids of 300 fit."""
    pool = _SessionMemoryPool(1024)
    admitted = 0
    for _ in range(10):
        try:
            pool.admit("probe", 300)
        except Exception:
            continue
        admitted += 1
    return admitted


_IMPLEMENTATION_ROOTS = ("compiler", "runtime")


def source_touches_implementation(function, _depth=4) -> bool:
    """Say whether a measurement source actually reaches the implementation.

    A source that returned a literal would agree with its ADR no matter what
    the code said, which is the shape of a tautology this registry is built to
    refuse. Reaching `compiler` or `runtime` is the weakest honest evidence
    that the number came from somewhere real.

    Most sources are one-line lambdas over a helper defined here, so the walk
    follows a helper of this module inward instead of stopping at its name.
    """
    if _depth <= 0:
        return False
    codes = [function.__code__]
    globals_map = function.__globals__
    seen, helpers = set(), []
    while codes:
        code = codes.pop()
        if id(code) in seen:
            continue
        seen.add(id(code))
        for constant in code.co_consts:
            if hasattr(constant, "co_names"):
                codes.append(constant)
        for name in code.co_names:
            value = globals_map.get(name)
            if value is None:
                continue
            origin = getattr(value, "__module__", None) or getattr(value, "__name__", "")
            if isinstance(origin, str) and origin.split(".")[0] in _IMPLEMENTATION_ROOTS:
                return True
            if callable(value) and hasattr(value, "__code__") and origin == __name__:
                helpers.append(value)
    return any(source_touches_implementation(helper, _depth - 1) for helper in helpers)


def assert_source_touches_implementation(name, function):
    if not source_touches_implementation(function):
        raise AssertionError(
            f"Measurement source {name!r} never reaches compiler/ or runtime/: "
            "a source that cannot disagree with the ADR is not a source")
