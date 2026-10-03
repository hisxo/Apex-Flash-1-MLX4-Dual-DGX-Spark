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
    echo "missing config: $config_file (copy config.example.env first)" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$config_file"

required=(IMAGE CONTAINER_PREFIX MODEL_DIR CACHE_DIR MASTER_ADDR MASTER_PORT)
for variable in "${required[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
        echo "missing required setting: $variable" >&2
        exit 1
    fi
done
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "model directory does not exist: $MODEL_DIR" >&2
    exit 1
fi
if [[ "${DRAFTER_DIR:-none}" != none && ! -d "$DRAFTER_DIR" ]]; then
    echo "drafter directory does not exist: $DRAFTER_DIR" >&2
    exit 1
fi

mkdir -p "$CACHE_DIR"
container_name="${CONTAINER_PREFIX}-r${rank}"
if docker container inspect "$container_name" >/dev/null 2>&1; then
    echo "container already exists: $container_name" >&2
    echo "stop it explicitly with: $script_dir/stop-rank.sh $rank" >&2
    exit 1
fi

docker_options=(
    --detach
    --name "$container_name"
    --gpus all
    --ipc host
    --network host
    --shm-size "${SHM_SIZE:-16g}"
    --ulimit memlock=-1
    --ulimit stack=67108864
    --cap-add IPC_LOCK
    --volume "$MODEL_DIR:/model:ro"
    --volume "$CACHE_DIR:/cache"
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
    --env "TF_GLM_MTP=${TF_GLM_MTP:-1}"
    --env TENSORFOLD_NO_UPDATE_CHECK=1
    --env HF_HUB_OFFLINE=1
)

for variable in \
    TF_GLM_KV \
    TF_GLM_CACHE_GIB \
    TF_GLM_DRAFT_RING \
    TENSORFOLD_MEMORY_RESERVE_GIB \
    TENSORFOLD_GLM_IMAGE_TOKENS \
    TENSORFOLD_GLM_MAX_IMAGES \
    TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS; do
    if [[ -n "${!variable:-}" ]]; then
        docker_options+=(--env "$variable=${!variable}")
    fi
done

if [[ -d /dev/infiniband ]]; then
    docker_options+=(--device /dev/infiniband)
fi
for variable in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA; do
    if [[ -n "${!variable:-}" ]]; then
        docker_options+=(--env "$variable=${!variable}")
    fi
done

drafter_options=(--drafter none)
if [[ "${DRAFTER_DIR:-none}" != none ]]; then
    docker_options+=(--volume "$DRAFTER_DIR:/drafter:ro")
    drafter_options=(--drafter /drafter)
fi

endpoint_options=()
if [[ "$rank" == 0 ]]; then
    endpoint_options=(
        --name "${SERVED_MODEL:-apex-flash-1-abliterated}"
        --host "${API_HOST:-127.0.0.1}"
        --port "${API_PORT:-8000}"
    )
fi

vision_options=()
if [[ "${VISION:-0}" == 1 ]]; then
    vision_options=(--vision)
fi

docker run "${docker_options[@]}" "$IMAGE" serve /model \
    --backend cuda \
    --tp 2 \
    --rank "$rank" \
    --master "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --context "${CONTEXT:-262144}" \
    --parallel "${PARALLEL:-1}" \
    --max-tokens "${MAX_TOKENS:-32768}" \
    --reasoning-effort "${REASONING_EFFORT:-high}" \
    --no-update-check \
    "${drafter_options[@]}" \
    "${vision_options[@]}" \
    "${endpoint_options[@]}"

echo "started $container_name; inspect with: docker logs -f $container_name"
