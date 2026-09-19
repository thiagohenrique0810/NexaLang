"""Local model importers with bounded, standard-library-only tensor I/O."""
from .safetensors import SafeTensorCheckpoint, SafeTensorError, SafeTensorReader


def import_llama_checkpoint(source_dir, destination, *, group_size=32, block_rows=64):
    """Import the supported local Llama subset; load model tooling on demand."""
    from .llama import import_llama_checkpoint as convert
    return convert(source_dir, destination, group_size=group_size, block_rows=block_rows)


__all__ = ['SafeTensorCheckpoint', 'SafeTensorError', 'SafeTensorReader', 'import_llama_checkpoint']
