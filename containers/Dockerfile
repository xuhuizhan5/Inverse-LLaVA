# check=skip=InvalidDefaultArgInFrom
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG UV_VERSION=0.12.7
WORKDIR /opt/invllava
# TorchAO is optional in Transformers and outside this repository's runtime
# contract. Remove a preinstalled copy before resolving the project layer.
RUN if python -m pip show torchao >/dev/null 2>&1; then \
      python -m pip uninstall --root-user-action=ignore -y torchao; \
    fi
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"
COPY pyproject.toml uv.lock requirements.txt README.md ./
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY third_party/licenses ./third_party/licenses
COPY requirements ./requirements
COPY containers ./containers
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tools ./tools
COPY tests ./tests
RUN uv venv --system-site-packages .venv && \
    . .venv/bin/activate && \
    uv sync --active --frozen --no-dev --extra analysis --extra distributed \
      --extra tracking --extra dev \
      --no-install-package torch --inexact

ENV PATH="/opt/invllava/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/huggingface \
    TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions \
    VIRTUAL_ENV=/opt/invllava/.venv
