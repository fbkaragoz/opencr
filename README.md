# <img src="public/opencr-logo.png" width="40" valign="middle"> OpenCR

> High-performance OCR pipeline for Turkish, archival, and complex-layout documents — turning PDFs into HuggingFace-ready training datasets.

OpenCR is an end-to-end open-source pipeline that converts PDFs (especially Turkish text, archival material, and pages with complex layout) into clean Parquet datasets ready for LLM training and retrieval.

For Turkish documents, see: [README.tr.md](./README.tr.md)

---

## Why OpenCR?

- **Turkish-first accuracy.** Built around DeepSeek-OCR, it handles Turkish characters and difficult page layouts better than off-the-shelf OCR.
- **Dataset factory.** Outputs are packaged directly as `pages.parquet` + `documents.parquet` with deterministic train/validation/test splits and a HuggingFace dataset card.
- **Operator console.** A single-page web UI to monitor runs, page-by-page validate quality, retry, and publish to HuggingFace.
- **Pluggable backends.** Production-grade NVIDIA + vLLM by default; runs in-process on Apple Silicon / CPU for development; or talk to any OpenAI-compatible model server.

---

## Quickstart

### Option 1 — Docker (NVIDIA GPU, fastest path to inference)

Requires Docker, an NVIDIA GPU, and the NVIDIA Container Toolkit.

```bash
docker compose up -d
```

Open http://localhost:39672. Drop PDFs in `./input/`, hit **Start OCR run**.

### Option 2 — Local process inference (default local setup)

For local development, demos, and small jobs on a single machine.

```bash
git clone https://github.com/cdliai/opencr.git
cd opencr
python3 -m venv .venv && source .venv/bin/activate
pip install -r ocr_pipeline/requirements.txt -r requirements-local.txt
MODEL_BACKEND=local ./scripts/start.sh
```

### Option 2b — Linux/Windows with NVIDIA GPU (local in-process inference)

For local in-process inference on GPU, install CUDA-matched PyTorch first (do **not**
install the default CUDA wheels if your driver stack is not matching):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r ocr_pipeline/requirements.txt -r requirements-local.txt
MODEL_BACKEND=local LOCAL_DEVICE=cuda ./scripts/start.sh
```

Then verify CUDA is visible before processing:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

### Run with explicit backend

```bash
# Explicit local CPU run
LOCAL_DEVICE=cpu MODEL_BACKEND=local ./scripts/start.sh

# Explicit local CUDA run
LOCAL_DEVICE=cuda MODEL_BACKEND=local ./scripts/start.sh

# Remote backend (model server at MODEL_SERVER_URL)
MODEL_BACKEND=remote MODEL_SERVER_URL=https://your-endpoint MODEL_API_KEY=sk-... ./scripts/start.sh
```

Open http://localhost:39672. The DeepSeek-OCR model (~6 GB) downloads 
on first request and runs in-process via `transformers` on CPU, MPS (Apple Silicon),
or CUDA (if available). Expect **5–30 seconds per page on M-series, much slower on CPU** — 
fine for development, not for production batch jobs.

### Option 3 — Remote model server (point at any OpenAI-compatible endpoint)

If you already run vLLM somewhere, or use OpenRouter, or another endpoint 
serving DeepSeek-OCR:

```bash
pip install -r ocr_pipeline/requirements.txt
MODEL_BACKEND=remote MODEL_SERVER_URL=https://your-endpoint MODEL_API_KEY=sk-... ./scripts/start.sh
```

---

## Configuration

Configurable via environment variables (or a `.env` file):

| Variable             | Default                          | Description                                                                                       |
| -------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------- |
| `MODEL_BACKEND`      | `vllm`                           | `vllm` (NVIDIA, OpenAI-compatible server), `local` (in-process transformers), `remote` (alias).   |
| `MODEL_SERVER_URL`   | `http://ocr-model:39671`         | Base URL for `vllm` / `remote` backends.                                                          |
| `MODEL_NAME`         | `deepseek-ai/DeepSeek-OCR`       | Model identifier.                                                                                 |
| `MODEL_API_KEY`      | `EMPTY`                          | API key for remote endpoints.                                                                     |
| `LOCAL_DEVICE`       | auto                             | `auto`, `mps`, `cuda`, or `cpu` for the `local` backend.                                          |
| `INPUT_DIR`          | `./input` (or `/data/input`)     | Where to read PDFs from.                                                                          |
| `OUTPUT_DIR`         | `./output` (or `/data/output`)   | Where artifacts and the SQLite DB land.                                                           |
| `HOST` / `PORT`      | `0.0.0.0` / `39672`              | Where the web console serves.                                                                     |
| `HF_OAUTH_CLIENT_ID` | unset                            | Enables "Sign in with HuggingFace" for the publish flow. See [HF OAuth setup](#hf-oauth-optional).|
| `APP_SESSION_SECRET` | random per process               | Cookie-signing secret. Set to a stable value in production.                                       |

---

## HuggingFace publishing

Completed runs can be pushed to a HuggingFace dataset repo. Two modes:

1. **Paste-token (default).** In the operator console, click **Publish to HuggingFace** and paste a HF write token. Or set `HF_TOKEN` in the server's environment to skip pasting.
2. **Sign in with HuggingFace (recommended for shared deployments).** Configure OAuth (below) and users sign in with their HF account. The publish flow then uses their personal token automatically. This is also how the operator console gets gated — without a session, the publish action is hidden.

### HF OAuth (optional)

1. Create an OAuth app at https://huggingface.co/settings/connected-applications/new with redirect URI `https://your-host/api/auth/callback` and scopes `openid profile write-repos`.
2. Set on the server:

```bash
export HF_OAUTH_CLIENT_ID=...
export HF_OAUTH_CLIENT_SECRET=...
export HF_OAUTH_REDIRECT_URI=https://your-host/api/auth/callback
export APP_SESSION_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
```

3. Restart. The console gains a **Sign in with HuggingFace** button in the topbar.

Published datasets are tagged `opencr` so they're discoverable via [HuggingFace's tag search](https://huggingface.co/datasets?other=opencr).

---

## Architecture

```
                ┌───────────────────────────────┐
                │      OCR pipeline (FastAPI)   │
   PDFs ─────►. │  ingest → render → OCR →      │   ──►  pages.parquet
                │  clean → validate → export    │        documents.parquet
                │  + operator console (Alpine)  │        manifest.json
                └──────────────┬────────────────┘
                               │ OpenAI-compatible
                               ▼
                ┌───────────────────────────────┐
                │  Model backend                │
                │  ┌─────────────────────────┐  │
                │  │ vllm (NVIDIA, prod)     │  │
                │  │ local (MPS/CPU, dev)    │  │
                │  │ remote (any OpenAI URL) │  │
                │  └─────────────────────────┘  │
                └───────────────────────────────┘
```

State lives in SQLite + the filesystem. 
No external queue/broker is required for single-node operation. 
See [docs/architectural-overhaul-v2.md](./docs/architectural-overhaul-v2.md) for the long-form design.

---

## Development

```bash
make install         # create venv, install deps
make run             # start dev server with sensible defaults
make test            # run pytest suite
make lint            # ruff check
```

Tests live under `tests/`. UI is plain HTML + Alpine.js — no build step.

---

## Contributing

Contributions are welcome — bug reports, Turkish-language 
test fixtures, benchmarks against other OCR engines, model-backend 
ports (MLX, llama.cpp), and documentation translations are 
especially useful. 

See [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).

OpenCR is built and maintained by [cdli.ai](https://cdli.ai) to support Turkish-language LLM research and dataset curation.
