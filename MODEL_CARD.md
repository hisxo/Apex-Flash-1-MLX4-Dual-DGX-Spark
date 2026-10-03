---
base_model:
  - cantina-security/apex-flash-1-abliterated
license: mit
library_name: tensorfold
pipeline_tag: image-text-to-text
tags:
  - glm5-next
  - mlx
  - 4-bit
  - dgx-spark
  - vision
---

# Apex Flash 1 MLX4 for Dual DGX Spark

This checkpoint is an independent MLX affine 4-bit conversion of
[`cantina-security/apex-flash-1-abliterated`](https://huggingface.co/cantina-security/apex-flash-1-abliterated),
prepared for TensorFold tensor-parallel inference on two NVIDIA DGX Spark
systems.

[Apex Flash 1](https://www.cantina.security/apex-flash) is Cantina Security's
open-weights security model, developed in partnership with Yeta. Thank you to
both teams for making it available. This conversion is not affiliated with or
endorsed by Cantina Security or Yeta. All rights and licensing for the source
model remain with the upstream rights holders.

## Provenance

| Item | Revision |
| --- | --- |
| Source | `cecb5eeb9c6b32404a0dd930df81de2c239bdd84` |
| TensorFold layout reference | `d2b17f5253440224422b150705c1109a9a3a0368` |
| TensorFold runtime | `6ea5ade26c4335491c50275af0be32b81f75f525` (0.6.4) |

The layout reference supplied only the expected tensor-key structure. No
reference checkpoint weights were copied into this conversion.

## Format

- MLX affine 4-bit weights
- 64 input values per quantization group
- BF16 scales and biases
- approximately 169 GiB of tensor data
- source tokenizer, chat template and model configuration retained

Conversion is shard-wise and deterministic. The included verifier can check
the complete output structure, sampled packed values, or every converted value
against the source checkpoint.

## Intended runtime

- TensorFold 0.6.4 for the upstream text-only profile
- two DGX Spark nodes
- tensor parallelism 2
- OpenAI-compatible endpoint
- 262,144-token upstream text profile
- experimental patched CUDA vision profile with a 360,000-token configured
  context window

The checkpoint includes the source MTP layer. DFlash2 is optional, separately
licensed and not included.

## Limitations

The 4-bit conversion is lossy. It preserves source lineage, not guaranteed
BF16 behavioral equivalence. Cybersecurity or pwn reasoning quality has not
been independently established against the source checkpoint, and upstream
evaluation results must not be attributed to this conversion without a direct
comparison. The abliterated source itself is experimental and has not undergone
the standard checkpoint's separate full-suite evaluation.

TensorFold 0.6.4 does not support image input for this model on CUDA. The
separate vision profile uses a pinned patched TensorFold 0.5.0 runtime and is
not an upstream TensorFold feature. Its BF16 vision tower and 360,000-token
window passed startup plus text/image smoke tests; a prompt filled to the
maximum depth has not yet been qualified. No cyber-task or throughput claim is
made without a published benchmark receipt.

Use the model only for authorized activity and review the source model card and
license before deployment or redistribution.
