# Security

Never place API keys in YAML, result manifests, prediction files, containers, or
W&B configuration. Use environment variables or a platform secret manager.
Dataset archives are verified and extracted with traversal checks. Checkpoints
must use `safetensors` for model weights; loading arbitrary pickle-based weights
requires an explicit conversion/audit step in an isolated environment.

Report suspected credential exposure or unsafe artifact handling privately to
the repository maintainers.
