"""Every packed codec must describe itself, at any group size.

Found by running a 125.8M-parameter synthetic model: a Q2 bundle with
group_size 32 converts, packs and passes `inspect --verify`, then refuses to
open for execution. The session described every packed codec as Q4 storage, and
Q4's four-bits-per-value floor rejects Q2's three bits. The existing fixtures
all use group_size 4, where the four-byte scale inflates Q2 to ten bits per
value and the wrong floor is cleared by accident.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.model_ir import DType, TensorDesc
from test_dense_weights_regressions import _DenseFixture


# bits per stored value = codec bits + 32 / group_size, for the F32 group scale.
# At group 4 the scale dominates and every codec clears four bits; at 32 it does
# not, which is why only the larger group exposes the bug.
GROUP_BITS = {("q2", 4): 10.0, ("q2", 32): 3.0, ("q3", 4): 11.0, ("q3", 32): 4.0,
              ("q4", 4): 12.0, ("q4", 32): 5.0, ("q8", 4): 16.0, ("q8", 32): 9.0}


class PackedStorageContractRegressions(unittest.TestCase):
    def test_the_dtype_set_covers_every_codec_the_runtime_dispatches(self):
        from runtime.nexapack.transformer import _PACKED_KERNELS
        from compiler.model_ir import _PACKED_BITS, _WEIGHT_STORAGE
        dispatched = {name.split("_")[0].lower() for name in _PACKED_KERNELS}
        self.assertEqual(dispatched, {"q2", "q3", "q4", "q8"})
        for codec in dispatched:
            with self.subTest(codec=codec):
                # A codec the runtime can execute but the IR cannot name is a
                # tensor that converts and then refuses to open.
                dtype = DType(codec)
                self.assertIn(dtype, _PACKED_BITS)
                self.assertIn(dtype, _WEIGHT_STORAGE)
                self.assertEqual(_PACKED_BITS[dtype], int(codec[1:]))

    def test_a_tensor_declared_as_its_own_codec_accepts_its_real_payload(self):
        rows, cols = 64, 128
        for (codec, group), bits in sorted(GROUP_BITS.items()):
            with self.subTest(codec=codec, group=group):
                payload = (rows * cols * int(bits) + 7) // 8
                desc = TensorDesc("w", (rows, cols), storage_dtype=codec, storage_nbytes=payload)
                self.assertEqual(desc.storage_dtype, DType(codec))
                # Declared as q4, the same real payload is rejected below 4 bits.
                floor = (rows * cols * 4 + 7) // 8
                if payload < floor:
                    with self.assertRaises(ValueError):
                        TensorDesc("w", (rows, cols), storage_dtype="q4", storage_nbytes=payload)


class PackedStorageExecutionRegressions(unittest.TestCase):
    """A bundle whose matrices are wider than the group: where the bug is real.

    With hidden_size 8 and group_size 32 every group holds 8 values, so the
    four-byte scale inflates Q2 to six bits per value and the wrong Q4 floor is
    cleared by accident. Only a matrix at least as wide as the group exposes it.
    """

    def setUp(self):
        import random
        import tempfile
        from compiler.model_config import ModelConfig
        from runtime.nexapack.bundle import write_model_bundle
        self.directory = Path(tempfile.mkdtemp())
        self.config = ModelConfig(name="packed_storage_fixture", vocab_size=13,
                                  hidden_size=64, intermediate_size=96,
                                  num_hidden_layers=1, num_attention_heads=1,
                                  num_key_value_heads=1, max_position_embeddings=16,
                                  tie_word_embeddings=True, rms_norm_eps=1e-5,
                                  rope_theta=10000.0)
        self.bundles = {}
        for codec in ("q2", "q3", "q4", "q8"):
            rng = random.Random(411)

            def source(shape, rng=rng):
                if len(shape) == 1:
                    return iter([[rng.uniform(0.85, 1.15) for _ in range(shape[0])]])
                return ([rng.uniform(-0.15, 0.15) for _ in range(shape[1])]
                        for _ in range(shape[0]))

            shapes = self.config.required_tensor_shapes()
            sources = {name: (lambda shape=shape: source(shape)) for name, shape in shapes.items()}
            path = self.directory / f"bundle-{codec}"
            write_model_bundle(path, self.config, sources, group_size=32, block_rows=16,
                               tensor_codecs={name: codec for name, shape in shapes.items()
                                              if len(shape) == 2})
            self.bundles[codec] = path

    def tearDown(self):
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_every_codec_opens_for_execution_at_group_size_32(self):
        from runtime.nexapack.transformer import TransformerSession
        for codec, bundle in sorted(self.bundles.items()):
            with self.subTest(codec=codec, group_size=32):
                # Opening is where the storage contract is checked: a tensor
                # described as another codec converts and verifies cleanly, then
                # fails here. Q2 at group 32 stores 3 bits per value against a
                # Q4 floor of 4, so only Q2 actually failed before the fix.
                session = TransformerSession(bundle, memory_budget="64MiB",
                                             max_sequence_length=8, tile_rows=8)
                try:
                    logits = session.prefill([1, 3])
                    self.assertEqual(len(logits), 2)
                    self.assertEqual(len(logits[0]), self.config.vocab_size)
                finally:
                    session.close()

    def test_the_session_describes_each_tensor_with_its_own_codec(self):
        from runtime.nexapack.transformer import TransformerSession
        for codec, bundle in sorted(self.bundles.items()):
            with self.subTest(codec=codec):
                session = TransformerSession(bundle, memory_budget="64MiB",
                                             max_sequence_length=8, tile_rows=8)
                try:
                    declared = {desc.storage_dtype.value for name, desc
                                in session._storage.items()
                                if len(desc.shape) == 2}
                    self.assertEqual(declared, {codec})
                finally:
                    session.close()


if __name__ == "__main__":
    unittest.main()
