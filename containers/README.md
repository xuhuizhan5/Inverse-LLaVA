# Reproducible GPU environment

The container builds on the pinned PyTorch 2.13/CUDA 13.0 development image
in `base-image.digest`. It retains that image's CUDA-matched PyTorch, cuDNN,
and Triton, and installs the project dependencies from `uv.lock`.

From the repository root on a Linux x86-64 Docker host:

```bash
docker build \
  --build-arg BASE_IMAGE="$(cat containers/base-image.digest)" \
  -t inverse-llava:local -f containers/Dockerfile .
docker run --rm inverse-llava:local invllava --help
docker run --rm inverse-llava:local python scripts/verify_config_catalog.py
```

Use the existing NVIDIA driver on the host and NVIDIA Container Toolkit for GPU
access. Mount your data and outputs at `/workspace`; the image includes no
weights or datasets. Multi-worker training needs sufficient shared memory.
For example:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v /absolute/path/to/workspace:/workspace \
  inverse-llava:local python -c \
  'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
```

Then follow the [training](../docs/training.md) or
[evaluation](../docs/evaluation.md) guide inside the container. Run a short
save/resume and generation check before a long job on a new GPU stack. Record
the image identity, installed package versions, device, and driver with results.
Builds on ARM64 require a platform-compatible NVIDIA base and separate testing;
do not assume this x86-64 image runs natively on Jetson.

## Dependency handling

`BASE_IMAGE` has no default: builds must supply the immutable image reference
in `base-image.digest`. The root `.dockerignore` limits the build context to
source and runtime inputs, excluding credentials, data, and manuscript files.

PyTorch is supplied by the base image. The uv lock uses a CPU wheel for
dependency resolution, while `--no-install-package torch --inexact` preserves
the installed CUDA build. Check the resulting environment before experiments;
a successful dependency resolution alone does not verify GPU compatibility.

The image removes optional TorchAO, which this implementation does not require.
The optional compiled runtime uses the Triton version shipped with PyTorch.
Keep that pairing instead of installing a separate Triton wheel. See
[optional kernel optimization](../docs/guides/KERNEL_OPTIMIZATION.md) for
numerical checks and profiling.

No container registry or cloud-provider account is needed for a local build.
To share an image, publish it to a registry you control and record its immutable
digest alongside the source and experiment configuration.

The reproduction manifests use `containers/publication-image.digest` for the
built execution image. Create that local record with your immutable image
reference before an environment-specific scientific audit. It is distinct from
the base-image digest and is not prefilled with a placeholder.
