#!/usr/bin/env bash
set -euo pipefail

rank=${1:-}
if [[ "$rank" != 0 && "$rank" != 1 ]]; then
    echo "usage: $0 0|1" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_file=${CONFIG_FILE:-"$script_dir/config.env"}
if [[ ! -f "$config_file" ]]; then
    echo "missing config: $config_file" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$config_file"
container_name="${CONTAINER_PREFIX:?missing CONTAINER_PREFIX}-r${rank}"

if ! docker container inspect "$container_name" >/dev/null 2>&1; then
    echo "container does not exist: $container_name"
    exit 0
fi

docker stop "$container_name"
docker rm "$container_name"
echo "removed container $container_name; model and cache data were retained"
