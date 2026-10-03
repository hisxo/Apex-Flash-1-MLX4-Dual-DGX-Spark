ARG BASE_IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:6ee3c6e0430040b69ddcb0c96c7fbbcb94a5bed47d48a8ba092626369ae533b9
FROM ${BASE_IMAGE}

COPY patches/0001-accept-native-glm-media-template.patch /tmp/

RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch -p1 < /tmp/0001-accept-native-glm-media-template.patch \
    && python -m py_compile tensorfold/families/glm5_next/cuda/app.py \
    && rm /tmp/0001-accept-native-glm-media-template.patch

ENV TENSORFOLD_NO_UPDATE_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton \
    CUDA_CACHE_PATH=/cache/nv/ComputeCache \
    CUDA_CACHE_MAXSIZE=4294967296

ENTRYPOINT ["tensorfold"]
