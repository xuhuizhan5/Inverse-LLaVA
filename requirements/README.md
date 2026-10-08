# Dependency environments

The root `uv.lock` pins the Python dependency graph used by the x86/CUDA
environment. It resolves PyTorch through a CPU index; the
[container build](../containers/README.md) retains its CUDA-matched system
PyTorch using `--no-install-package torch --inexact`. Record the installed
package inventory, CUDA runtime, and driver for each execution environment.
The source identity includes the lock.

`requirements.txt` provides the pinned project dependencies for installation
over an existing compatible GPU environment. Follow its comments and preserve
the platform's CUDA-matched PyTorch. ARM64 systems such as Jetson need a
platform-compatible NVIDIA image and their own functional checks.

`lmms-eval.in` defines the separate 0.7.1 reference-evaluator environment.
Resolve it to `lmms-eval-lock.txt` with hashes on the execution host, then use
`scripts/bootstrap_lmms_golden.sh`. This optional environment supplies
differential scorer checks; native training and evaluation do not require it.
Keep raw predictions and the pinned scorer identity with each comparison.

Triton is supplied by the CUDA PyTorch build. The optional TorchInductor path
uses that matched version; the default SDPA path does not import Triton directly.
