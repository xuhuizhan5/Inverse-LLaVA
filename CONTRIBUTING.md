# Contributing

Changes must preserve the distinction between scientific configuration
(architecture, data, seed, protocol), execution configuration (hardware,
workers, output paths), and generated artifacts. New benchmark support requires
a benchmark card, a frozen protocol configuration, a tiny fixture, and a scorer
test before a publication result is accepted.

Do not commit weights, raw datasets, credentials, caches, or generated run
directories. Include small redistributable test fixtures, YAML configurations,
and documentation assets. Keep manuscript sources, reviewer correspondence,
and private experiment records outside the public repository.

Before review, run:

```bash
./scripts/lint.sh
python -m mypy src
pytest -m "not gpu and not network and not slow"
```

`scripts/validate_static.sh` is the dependency-light syntax, catalog, and link
check; it is useful on constrained machines but is not a substitute for the
three commands above.
