#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 SHARD_INDEX SHARD_COUNT DESTINATION" >&2
}

if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi

shard_index=$1
shard_count=$2
destination=$3

if ! [[ "$shard_index" =~ ^[0-9]+$ && "$shard_count" =~ ^[1-9][0-9]*$ ]]; then
    usage
    exit 2
fi
if (( shard_index >= shard_count )); then
    echo "SHARD_INDEX must be smaller than SHARD_COUNT" >&2
    exit 2
fi

hf_bin=${HF_BIN:-hf}
source_repo=cantina-security/apex-flash-1-abliterated
source_revision=cecb5eeb9c6b32404a0dd930df81de2c239bdd84
total_shards=62

command -v "$hf_bin" >/dev/null || {
    echo "Hugging Face CLI not found: $hf_bin" >&2
    exit 1
}

files=(
    .gitattributes
    LICENSE
    README.md
    chat_template.jinja
    config.json
    generation_config.json
    model.safetensors.index.json
    processor_config.json
    tokenizer.json
    tokenizer_config.json
)

for ((number = shard_index + 1; number <= total_shards; number += shard_count)); do
    printf -v shard "model-%05d-of-%05d.safetensors" "$number" "$total_shards"
    files+=("$shard")
done

echo "downloading partition $shard_index/$shard_count (${#files[@]} files)"
exec "$hf_bin" download "$source_repo" "${files[@]}" \
    --revision "$source_revision" \
    --local-dir "$destination"
