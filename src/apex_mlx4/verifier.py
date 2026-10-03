"""Verify the structure and conversion values of an Apex Flash MLX checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from apex_mlx4.converter import (
    DTYPE_BYTES,
    quantize_affine4,
    read_weight_map,
    validate_layout,
)


def bytes_for(handle, name: str) -> int:
    view = handle.get_slice(name)
    size = DTYPE_BYTES[str(view.get_dtype())]
    for dimension in view.get_shape():
        size *= int(dimension)
    return size


def unpack_group(words: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, dtype=torch.int64) * 4
    unsigned = words.to(torch.int64) & 0xFFFFFFFF
    return ((unsigned[..., None] >> shifts) & 0xF).reshape(-1)


def _assert_equal(left: torch.Tensor, right: torch.Tensor, *, label: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
        raise ValueError(f"tensor mismatch: {label}")


def verify_preserved_tensor(source, output, name: str, row_chunk: int) -> None:
    shape = source.get_slice(name).get_shape()
    if not shape:
        _assert_equal(source.get_tensor(name), output.get_tensor(name), label=name)
        return
    for start in range(0, int(shape[0]), row_chunk):
        stop = min(start + row_chunk, int(shape[0]))
        _assert_equal(
            source.get_slice(name)[start:stop],
            output.get_slice(name)[start:stop],
            label=f"{name}[{start}:{stop}]",
        )


def verify_sampled_shard(
    source_file: Path,
    output_file: Path,
    quantized: set[str],
    *,
    row_chunk: int,
) -> tuple[int, int]:
    quant_checks = 0
    preserved_checks = 0
    with (
        safe_open(str(source_file), framework="pt", device="cpu") as source,
        safe_open(str(output_file), framework="pt", device="cpu") as output,
    ):
        quant_names = [name for name in source.keys() if name in quantized]
        preserved_names = [name for name in source.keys() if name not in quantized]

        if quant_names:
            name = quant_names[len(quant_names) // 2]
            stem = name.removesuffix(".weight")
            rows, width = source.get_slice(name).get_shape()
            for row in sorted({0, rows // 2, rows - 1}):
                source_row = source.get_slice(name)[row : row + 1].float().reshape(-1)
                packed_row = output.get_slice(name)[row : row + 1]
                scale_row = output.get_slice(stem + ".scales")[row : row + 1].float()
                bias_row = output.get_slice(stem + ".biases")[row : row + 1].float()
                for group in sorted({0, width // 128, width // 64 - 1}):
                    values = source_row[group * 64 : (group + 1) * 64]
                    scale = scale_row[0, group]
                    bias = bias_row[0, group]
                    expected = torch.round((values - bias) / scale).clamp(0, 15)
                    expected = expected.to(torch.int64)
                    actual = unpack_group(packed_row[0, group * 8 : (group + 1) * 8])
                    if not torch.equal(actual, expected):
                        raise ValueError(
                            f"quantization mismatch: {output_file.name}:{name} "
                            f"row={row} group={group}"
                        )
                    quant_checks += 1

        if preserved_names:
            name = preserved_names[len(preserved_names) // 2]
            verify_preserved_tensor(source, output, name, row_chunk)
            preserved_checks += 1
    return quant_checks, preserved_checks


def verify_full_shard(
    source_file: Path,
    output_file: Path,
    quantized: set[str],
    *,
    device: torch.device,
    row_chunk: int,
) -> tuple[int, int]:
    quant_checks = 0
    preserved_checks = 0
    with (
        safe_open(str(source_file), framework="pt", device="cpu") as source,
        safe_open(str(output_file), framework="pt", device="cpu") as output,
    ):
        for name in source.keys():
            if name not in quantized:
                verify_preserved_tensor(source, output, name, row_chunk)
                preserved_checks += 1
                continue

            stem = name.removesuffix(".weight")
            rows = int(source.get_slice(name).get_shape()[0])
            for start in range(0, rows, row_chunk):
                stop = min(start + row_chunk, rows)
                expected = quantize_affine4(
                    source.get_slice(name)[start:stop],
                    device=device,
                    row_chunk=row_chunk,
                )
                actual = (
                    output.get_slice(name)[start:stop],
                    output.get_slice(stem + ".scales")[start:stop],
                    output.get_slice(stem + ".biases")[start:stop],
                )
                for suffix, expected_part, actual_part in zip(
                    ("weight", "scales", "biases"), expected, actual, strict=True
                ):
                    _assert_equal(
                        expected_part,
                        actual_part,
                        label=f"{name}:{suffix}[{start}:{stop}]",
                    )
            quant_checks += 1
    return quant_checks, preserved_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--template-index", type=Path, required=True)
    parser.add_argument("--level", choices=("structure", "sample", "full"), default="sample")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.row_chunk < 1:
        raise ValueError("--row-chunk must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("need 0 <= --shard-index < --shard-count")

    source_map = read_weight_map(args.source / "model.safetensors.index.json")
    template_map = read_weight_map(args.template_index)
    output_map = read_weight_map(args.model / "model.safetensors.index.json")
    quantized = validate_layout(source_map, template_map)
    if set(output_map) != set(template_map):
        raise ValueError("converted and reference tensor sets differ")

    total_size = 0
    found: dict[str, str] = {}
    shard_names = sorted(set(source_map.values()))
    for filename in shard_names:
        path = args.model / filename
        if not path.exists():
            raise FileNotFoundError(path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in found:
                    raise ValueError(f"duplicate tensor {name}")
                found[name] = filename
                total_size += bytes_for(handle, name)
    if found != output_map:
        raise ValueError("model index does not match shard contents")

    quant_checks = 0
    preserved_checks = 0
    local_shards = 0
    device = torch.device(args.device)
    if args.level == "full" and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    if args.level != "structure":
        for position, filename in enumerate(shard_names):
            if position % args.shard_count != args.shard_index:
                continue
            source_file = args.source / filename
            if not source_file.exists():
                raise FileNotFoundError(source_file)
            if args.level == "full":
                quant_count, preserved_count = verify_full_shard(
                    source_file,
                    args.model / filename,
                    quantized,
                    device=device,
                    row_chunk=args.row_chunk,
                )
            else:
                quant_count, preserved_count = verify_sampled_shard(
                    source_file,
                    args.model / filename,
                    quantized,
                    row_chunk=args.row_chunk,
                )
            quant_checks += quant_count
            preserved_checks += preserved_count
            local_shards += 1

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    expected_quantization = {"bits": 4, "group_size": 64, "mode": "affine"}
    if (
        config.get("quantization") != expected_quantization
        or config.get("quantization_config") != expected_quantization
    ):
        raise ValueError("config.json has incorrect quantization metadata")

    model_index = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    indexed_size = int(model_index["metadata"]["total_size"])
    if total_size != indexed_size:
        raise ValueError(f"index total_size={indexed_size}, actual={total_size}")

    detail = "structure only"
    if args.level != "structure":
        detail = (
            f"{args.level} value checks: {quant_checks} quantized + "
            f"{preserved_checks} preserved across {local_shards} local source shards"
        )
    print(
        f"ok: {len(found)} tensors, {len(shard_names)} shards, "
        f"{total_size / 2**30:.2f} GiB; {detail}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
