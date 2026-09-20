#!/usr/bin/env python3
"""Pack a model bundle directory into one `.nxb`, unpack it, or verify it.

Packing copies the bundle's files verbatim; nothing is re-encoded, so the
tensors inside a container are the same bytes the directory held. The reported
numbers are measured from the written file, never estimated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.nexapack.container import NexaContainerReader, pack_bundle, unpack_bundle


def _report(measurements):
    print(json.dumps(measurements, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packer = commands.add_parser("pack", help="Write a .nxb from a bundle directory")
    packer.add_argument("bundle", type=Path)
    packer.add_argument("output", type=Path)
    unpacker = commands.add_parser("unpack", help="Rebuild the bundle directory from a .nxb")
    unpacker.add_argument("container", type=Path)
    unpacker.add_argument("output", type=Path)
    verifier = commands.add_parser("verify", help="Read every section and check its checksum")
    verifier.add_argument("container", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            _report(pack_bundle(args.bundle, args.output))
        elif args.command == "unpack":
            _report(unpack_bundle(args.container, args.output))
        else:
            with NexaContainerReader(args.container) as container:
                measurements = container.measurements()
                measurements["verified_bytes"] = container.verify()
                measurements["checksums_verified"] = True
                _report(measurements)
        return 0
    except (OSError, ValueError, ArithmeticError, MemoryError) as exc:
        print(f"Nexa packaging failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
