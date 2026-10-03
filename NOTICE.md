# Third-party material

This repository contains conversion and deployment code only. It does not
redistribute model weights.

- Source checkpoint: [`cantina-security/apex-flash-1-abliterated`](https://huggingface.co/cantina-security/apex-flash-1-abliterated), MIT license. Cantina Security describes Apex Flash 1 as developed in partnership with Yeta and based on GLM-5.3-Flash.
- Layout reference: [`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP). The converter reads its tensor index as a layout manifest; it does not copy reference weights.
- Serving runtime: [TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0 license.
- Experimental vision runtime: the public TensorFold 0.5.0 image from
  [Mia AI Lab's dual-DGX GLM project](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
  Apache-2.0, pinned by digest in `deploy/vision-360k/Dockerfile`. The image
  contains NVIDIA's PyTorch container and other third-party software under
  their own terms. This repository adds only the media-template compatibility
  patch in that directory.
- Optional drafter: [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), CC BY-NC-ND 4.0. It is not included or downloaded by this repository.

Review the terms of every checkpoint before downloading, converting or
redistributing it. The repository's MIT license applies only to the code here.
