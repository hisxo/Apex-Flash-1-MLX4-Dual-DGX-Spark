from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from apex_mlx4.converter import (
    convert_shard,
    copy_metadata,
    finalize_index,
    quantize_affine4,
    read_weight_map,
    validate_layout,
)
from apex_mlx4.verifier import unpack_group, verify_full_shard


class AffineQuantizationTests(unittest.TestCase):
    def test_packing_shapes_and_error_bound(self) -> None:
        generator = torch.Generator().manual_seed(53)
        weight = torch.randn((19, 192), generator=generator, dtype=torch.bfloat16)
        weight[0, :64] = 0.25

        words, scales, biases = quantize_affine4(weight, device=torch.device("cpu"), row_chunk=7)

        self.assertEqual((words.dtype, words.shape), (torch.uint32, (19, 24)))
        self.assertEqual((scales.dtype, scales.shape), (torch.bfloat16, (19, 3)))
        self.assertEqual((biases.dtype, biases.shape), (torch.bfloat16, (19, 3)))

        unpacked = torch.stack([unpack_group(row) for row in words])
        groups = weight.float().reshape(19, 3, 64)
        expected = torch.round(
            (groups - biases.float().unsqueeze(-1)) / scales.float().unsqueeze(-1)
        )
        expected = expected.clamp(0, 15).to(torch.int64).reshape(19, 192)
        self.assertTrue(torch.equal(unpacked, expected))

        reconstructed = unpacked.reshape(19, 3, 64).float() * scales.float().unsqueeze(
            -1
        ) + biases.float().unsqueeze(-1)
        error = (reconstructed - groups).abs()
        bound = scales.float().unsqueeze(-1) * 0.5001 + 1e-6
        self.assertTrue(bool((error <= bound).all()))

    def test_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D"):
            quantize_affine4(torch.zeros(64), device=torch.device("cpu"), row_chunk=1)
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            quantize_affine4(torch.zeros((2, 65)), device=torch.device("cpu"), row_chunk=1)


class EndToEndTests(unittest.TestCase):
    def test_convert_resume_finalize_and_full_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            shard = "model-00001-of-00001.safetensors"

            generator = torch.Generator().manual_seed(7)
            tensors = {
                "layer.weight": torch.randn((5, 128), generator=generator, dtype=torch.bfloat16),
                "norm.weight": torch.randn(16, generator=generator, dtype=torch.bfloat16),
            }
            save_file(tensors, str(source / shard), metadata={"format": "pt"})
            source_map = {name: shard for name in tensors}
            template_map = {
                "layer.weight": shard,
                "layer.scales": shard,
                "layer.biases": shard,
                "norm.weight": shard,
            }
            (source / "config.json").write_text('{"model_type":"glm5_next"}\n')
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": source_map})
            )

            quantized = validate_layout(source_map, template_map)
            copy_metadata(
                source,
                output,
                source_repo="example/source",
                source_revision="source-revision",
                layout_repo="example/layout",
                layout_revision="layout-revision",
            )
            changed, tensor_count = convert_shard(
                source / shard,
                output / shard,
                quantized,
                device=torch.device("cpu"),
                row_chunk=2,
            )
            self.assertEqual((changed, tensor_count), (1, 4))
            self.assertEqual(
                convert_shard(
                    source / shard,
                    output / shard,
                    quantized,
                    device=torch.device("cpu"),
                    row_chunk=2,
                )[0],
                0,
            )

            index = finalize_index(source_map, output)
            self.assertEqual(set(index["weight_map"]), set(template_map))
            self.assertEqual(
                verify_full_shard(
                    source / shard,
                    output / shard,
                    quantized,
                    device=torch.device("cpu"),
                    row_chunk=2,
                ),
                (1, 1),
            )

    def test_index_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index.json"
            index.write_text(json.dumps({"weight_map": {"weight": "../outside"}}))
            with self.assertRaisesRegex(ValueError, "unsafe shard path"):
                read_weight_map(index)


if __name__ == "__main__":
    unittest.main()
