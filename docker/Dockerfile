ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}

ARG TENSORFOLD_COMMIT=6ea5ade26c4335491c50275af0be32b81f75f525

RUN python -m pip install --no-cache-dir \
      "git+https://github.com/ashhart/TensorFold.git@${TENSORFOLD_COMMIT}" \
    && tensorfold --version

ENV TENSORFOLD_NO_UPDATE_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton \
    CUDA_CACHE_PATH=/cache/nv/ComputeCache \
    CUDA_CACHE_MAXSIZE=4294967296

ENTRYPOINT ["tensorfold"]
