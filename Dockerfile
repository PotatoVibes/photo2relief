FROM python:3.12-slim

# Torch is installed via a mutually-exclusive uv extra. The base image uses the CPU
# wheels only (hard rule: no CUDA torch in the base image); the GPU image passes
# TORCH_EXTRA=cu128 via docker-compose.gpu.yml.
ARG TORCH_EXTRA=cpu

# libheif for pillow-heif; libgl/glib for opencv (added in M4) kept out until needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libheif1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /srv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra ${TORCH_EXTRA}

COPY app ./app
COPY static ./static
RUN uv sync --frozen --extra ${TORCH_EXTRA}

# DA3 worker: an isolated venv (depth_anything_3 pins numpy<2). GPU image only --
# the CPU image's default model is da2-small and stays slim; selecting da3mono-large
# there yields a clear BackendUnavailableError with setup instructions.
COPY da3worker ./da3worker
RUN if [ "${TORCH_EXTRA}" = "cu128" ]; then \
        cd da3worker && uv sync --frozen --extra cu128; \
    fi

ENV P2R_DATA_DIR=/srv/data
VOLUME ["/srv/data"]

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD uv run python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/api/health')" || exit 1

CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${P2R_PORT:-8090}"]
