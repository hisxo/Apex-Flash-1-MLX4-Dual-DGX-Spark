# Experimental vision + 360K profile

This profile serves the same converted Cantina checkpoint with GLM's BF16
vision tower and a 360,000-token prompt-plus-reply window on two DGX Sparks.
It is separate from the upstream TensorFold 0.6.4 text-only profile.

## Runtime

The image is pinned to the public TensorFold 0.5.0 CUDA vision runtime produced
by [Mia AI Lab's dual-DGX project](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
That runtime contains the GLM CUDA vision implementation; current upstream
TensorFold does not enable GLM image input on CUDA. The local patch only makes
the frontend accept Cantina's existing native image/video branch instead of
requiring the older upstream no-media reminder. It does not replace the
checkpoint's chat-template logic.

The base image is pinned by digest. Review its Apache-2.0 notice and NVIDIA
container terms before use.

## Qualified configuration

- MLX affine Q4/group-64 target weights
- BF16 latent and indexer caches; no FP8 cache fallback
- BF16 vision tower, 2,048 visual tokens per image
- DFlash2 drafting; the resident MTP target head is omitted
- one active request
- 360,000-token prompt-plus-reply window

Observed startup admission on the tested pair:

| Rank | Startup estimate | Budget |
| ---: | ---: | ---: |
| 0, vision | 92.24 GiB | 99.21 GiB |
| 1 | 91.10 GiB | 99.71 GiB |

After text and image smoke tests, `MemAvailable` was approximately 19 GiB on
rank 0 and 21 GiB on rank 1. The image test encoded 285 visual rows and read
`CORNER CAFE | 22.50` exactly from a synthetic receipt. These are deployment
validation results, not a full-depth quality or throughput benchmark.

## Build and start

From the repository root, build the same image on both nodes:

```bash
docker build -t apex-flash-tensorfold:vision-360k \
  -f deploy/vision-360k/Dockerfile deploy/vision-360k
```

Create the generic deployment configuration on each node:

```bash
cp deploy/vision-360k/config.example.env deploy/config.env
```

Set only local paths and private-link interface values. Start rank 1 before
rank 0:

```bash
./deploy/start-rank.sh 1
./deploy/start-rank.sh 0
```

The first start compiles CUDA extensions. Keep a separate `CACHE_DIR` from the
upstream 0.6.4 profile so both deployments remain independently recoverable.

## Scope

Admission, startup, text generation and image encoding are validated. A prompt
actually filled to 360K and cyber-task quality at that depth are not yet
claimed. Video is implemented by the pinned runtime but is not part of this
qualification receipt.
