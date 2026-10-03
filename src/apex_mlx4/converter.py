"""Convert a GLM-5.3-Flash checkpoint to TensorFold's MLX affine 4-bit layout.

The TensorFold checkpoint is used only as a tensor-layout manifest. Tensors
with sibling ``.scales`` and ``.biases`` entries in that manifest are
quantized; all other tensors retain their original dtype and values.

Conversion is shard-wise, resumable and partitionable across machines.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from apex_mlx4 import __version__

SOURCE_REPO = "cantina-security/apex-flash-1-abliterated"
SOURCE_REVISION = "cecb5eeb9c6b32404a0dd930df81de2c239bdd84"
LAYOUT_REPO = "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP"
LAYOUT_REVISION = "d2b17f5253440224422b150705c1109a9a3a0368"

SMALL_FILES = (
    "LICENSE",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _safe_relative_file(value: str, *, index_path: Path) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"{index_path}: unsafe shard path {value!r}")
    return value


def read_weight_map(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{path}: missing non-empty weight_map")
    return {
        str(name): _safe_relative_file(str(filename), index_path=path)
        for name, filename in weight_map.items()
    }


def quantized_weights(template_names: Iterable[str]) -> set[str]:
    names = set(template_names)
    result: set[str] = set()
    for name in names:
        if not name.endswith(".weight"):
            continue
        stem = name.removesuffix(".weight")
        if stem + ".scales" in names and stem + ".biases" in names:
            result.add(name)
    return result


def output_names(source_name: str, quantized: set[str]) -> tuple[str, ...]:
    if source_name not in quantized:
        return (source_name,)
    stem = source_name.removesuffix(".weight")
    return source_name, stem + ".scales", stem + ".biases"


def quantize_affine4(
    weight: torch.Tensor,
    *,
    device: torch.device,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2-D tensor into affine 4-bit groups of 64 input values."""

    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")

    rows, width = (int(value) for value in weight.shape)
    if width % 64:
        raise ValueError(f"input width must be divisible by 64, got {tuple(weight.shape)}")

    words = torch.empty((rows, width // 8), dtype=torch.uint32, device="cpu")
    scales = torch.empty((rows, width // 64), dtype=torch.bfloat16, device="cpu")
    biases = torch.empty_like(scales)

    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        groups = weight[start:stop].to(device=device, dtype=torch.float32)
        groups = groups.reshape(-1, width // 64, 64)
        if not torch.isfinite(groups).all():
            raise ValueError(f"non-finite source values in rows {start}:{stop}")

        low = groups.amin(dim=-1)
        high = groups.amax(dim=-1)
        scale = ((high - low) / 15).clamp_min(1e-8).to(torch.bfloat16)
        bias = low.to(torch.bfloat16)
        quantized = torch.round((groups - bias.float().unsqueeze(-1)) / scale.float().unsqueeze(-1))
        quantized = quantized.clamp_(0, 15).to(torch.int32)
        quantized = quantized.reshape(stop - start, width // 8, 8)

        packed = torch.zeros(quantized.shape[:2], dtype=torch.int32, device=device)
        for nibble in range(8):
            packed.bitwise_or_(quantized[..., nibble] << (4 * nibble))

        words[start:stop].copy_(packed.cpu().view(torch.uint32))
        scales[start:stop].copy_(scale.cpu())
        biases[start:stop].copy_(bias.cpu())

    return words, scales, biases


def keys_in_file(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def expected_file_keys(source_path: Path, quantized: set[str]) -> set[str]:
    with safe_open(str(source_path), framework="pt", device="cpu") as handle:
        return {
            output_name
            for source_name in handle.keys()
            for output_name in output_names(source_name, quantized)
        }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy_metadata(
    source_dir: Path,
    output_dir: Path,
    *,
    source_repo: str,
    source_revision: str,
    layout_repo: str,
    layout_revision: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        source = source_dir / name
        if source.exists():
            shutil.copyfile(source, output_dir / name)

    source_readme = source_dir / "README.md"
    if source_readme.exists():
        shutil.copyfile(source_readme, output_dir / "SOURCE_README.md")

    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    quantization = {"bits": 4, "group_size": 64, "mode": "affine"}
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    _write_json(output_dir / "config.json", config)

    provenance = {
        "converter": {
            "name": "apex-flash-1-mlx4-dual-dgx-spark",
            "version": __version__,
        },
        "layout_reference": {"repository": layout_repo, "revision": layout_revision},
        "quantization": quantization,
        "source": {"repository": source_repo, "revision": source_revision},
    }
    _write_json(output_dir / "conversion.json", provenance)


def validate_layout(source_map: dict[str, str], template_map: dict[str, str]) -> set[str]:
    quantized = quantized_weights(template_map)
    if not quantized:
        raise ValueError("reference layout contains no affine quantized weights")
    missing_sources = sorted(name for name in quantized if name not in source_map)
    if missing_sources:
        preview = "\n  ".join(missing_sources[:20])
        raise ValueError(f"layout has quantized weights absent from source:\n  {preview}")

    generated = {
        output_name
        for source_name in source_map
        for output_name in output_names(source_name, quantized)
    }
    template_names = set(template_map)
    if generated != template_names:
        missing = sorted(template_names - generated)
        extra = sorted(generated - template_names)
        raise ValueError(
            f"source and reference tensor layouts differ: missing={missing[:20]} extra={extra[:20]}"
        )
    return quantized


def convert_shard(
    source_path: Path,
    output_path: Path,
    quantized: set[str],
    *,
    device: torch.device,
    row_chunk: int,
) -> tuple[int, int]:
    expected = expected_file_keys(source_path, quantized)
    if output_path.exists() and keys_in_file(output_path) == expected:
        print(f"skip {output_path.name}: complete", flush=True)
        return 0, len(expected)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(source_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        source_names = list(handle.keys())
        for position, name in enumerate(source_names, 1):
            tensor = handle.get_tensor(name)
            if name in quantized:
                words, scales, biases = quantize_affine4(tensor, device=device, row_chunk=row_chunk)
                stem = name.removesuffix(".weight")
                tensors[name] = words
                tensors[stem + ".scales"] = scales
                tensors[stem + ".biases"] = biases
                action = "q4"
            else:
                tensors[name] = tensor.contiguous()
                action = str(tensor.dtype).removeprefix("torch.")
            print(
                f"{source_path.name} {position}/{len(source_names)} "
                f"{action} {name} {tuple(tensor.shape)}",
                flush=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, output_path)
    return 1, len(expected)


def finalize_index(source_map: dict[str, str], output_dir: Path) -> dict[str, object]:
    output_map: dict[str, str] = {}
    total_size = 0
    for filename in sorted(set(source_map.values())):
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing converted shard: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in output_map:
                    raise ValueError(f"duplicate output tensor: {name}")
                view = handle.get_slice(name)
                shape = view.get_shape()
                dtype = str(view.get_dtype())
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"unsupported dtype {dtype} for {name}")
                count = 1
                for dimension in shape:
                    count *= int(dimension)
                total_size += count * DTYPE_BYTES[dtype]
                output_map[name] = filename

    index = {"metadata": {"total_size": total_size}, "weight_map": output_map}
    _write_json(output_dir / "model.safetensors.index.json", index)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template-index", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--layout-repo", default=LAYOUT_REPO)
    parser.add_argument("--layout-revision", default=LAYOUT_REVISION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.row_chunk < 1:
        raise ValueError("--row-chunk must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("need 0 <= --shard-index < --shard-count")
    if args.source.resolve() == args.output.resolve():
        raise ValueError("--source and --output must be different directories")

    source_map = read_weight_map(args.source / "model.safetensors.index.json")
    template_map = read_weight_map(args.template_index)
    quantized = validate_layout(source_map, template_map)
    shard_names = sorted(set(source_map.values()))
    assigned = [
        name
        for index, name in enumerate(shard_names)
        if index % args.shard_count == args.shard_index
    ]
    print(
        f"layout ok: {len(source_map)} source tensors, {len(quantized)} quantized weights, "
        f"{len(template_map)} output tensors, {len(assigned)}/{len(shard_names)} assigned shards",
        flush=True,
    )
    if args.dry_run:
        return 0

    copy_metadata(
        args.source,
        args.output,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        layout_repo=args.layout_repo,
        layout_revision=args.layout_revision,
    )
    if not args.finalize_only:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        converted = 0
        for position, filename in enumerate(assigned, 1):
            print(f"shard {position}/{len(assigned)}: {filename}", flush=True)
            changed, _ = convert_shard(
                args.source / filename,
                args.output / filename,
                quantized,
                device=device,
                row_chunk=args.row_chunk,
            )
            converted += changed
        print(
            f"converted {converted} shard(s); complete existing shards were retained",
            flush=True,
        )

    if args.shard_count == 1 or args.finalize_only:
        index = finalize_index(source_map, args.output)
        if set(index["weight_map"]) != set(template_map):
            raise ValueError("final output tensor set differs from reference layout")
        size_gib = int(index["metadata"]["total_size"]) / 2**30
        print(f"finalized {len(index['weight_map'])} tensors, {size_gib:.2f} GiB", flush=True)
    else:
        print("partition complete; merge all shards and run --finalize-only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
