# local-model

A CLI tool for managing local LLM inference servers. Register GGUF models, start/stop servers, run benchmarks and quality tests — all from one command.

Works with any llama.cpp-compatible server binary (upstream llama.cpp, [PrismML](https://github.com/nicebread-cloud/prism-ml), [TurboQuant+](https://github.com/TheTom/turboquant_plus), etc.).

## Install

```bash
# From GitHub
pip install git+https://github.com/shakclaw-bot/local-model-cli.git

# Or clone and install locally
git clone https://github.com/shakclaw-bot/local-model-cli.git
cd local-model-cli
pip install -e .
```

## Quick Start: Ternary Bonsai 8B

[Ternary Bonsai 8B](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf) is a 1.58-bit quantized 8B model that runs in just **2.2 GB of RAM** — ideal for constrained machines. It uses PrismML's Q2_0 ternary format ({-1, 0, +1} weights with FP16 group-wise scaling), based on Qwen3-8B with 65K native context.

### 1. Build PrismML's llama-server

Ternary Bonsai requires PrismML's fork of llama.cpp for the Q2_0 quantization type:

```bash
git clone https://github.com/nicebread-cloud/prism-ml.git
cd prism-ml
mkdir build && cd build

# macOS (Apple Silicon — Metal acceleration)
cmake .. -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j

# Linux / Windows WSL (CPU with AVX-512 if available)
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j

# Windows (MSVC)
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 2. Configure the backend

```bash
# Point local-model to your PrismML llama-server binary
local-model config --set-backend default ./build/bin/llama-server
```

### 3. Download and register the model

```bash
local-model add hf:prism-ml/Ternary-Bonsai-8B-gguf bonsai
```

This downloads the GGUF from Hugging Face into `~/.local-model/models/` and registers it.

### 4. Tune the model settings (optional)

Edit `~/.local-model/registry.json` to set context window and KV cache:

```json
{
  "bonsai": {
    "name": "Ternary Bonsai 8B",
    "file": "Ternary-Bonsai-8B-Q2_0.gguf",
    "binary": "default",
    "port": 8080,
    "context": 65536,
    "cache_k": "f16",
    "cache_v": "f16",
    "flash_attn": "on",
    "threads": 4,
    "notes": "PrismML Q2_0 ternary. 2.18GB for 8B params. 65K native context."
  }
}
```

### 5. Start and use

```bash
local-model start bonsai
# Starting Ternary Bonsai 8B...
#   port: 8080  ctx: 65536  KV: f16/f16
#   waiting for health... ready (3s)

# The server is now listening at http://127.0.0.1:8080
# Compatible with any OpenAI-format client:
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai","messages":[{"role":"user","content":"Hello!"}]}'
```

### 6. Benchmark

```bash
local-model test bonsai    # Quality tests (reasoning, coding, factual, creative)
local-model bench bonsai   # Speed benchmark at multiple context sizes
```

## Commands

| Command | Description |
|---------|-------------|
| `local-model list` | Show registered models with status and bench speeds |
| `local-model start <model> [--ctx N]` | Start a model server |
| `local-model stop <model\|all>` | Stop running server(s) |
| `local-model status` | Show running servers with health and slot info |
| `local-model test <model>` | Run quality tests (reasoning, coding, factual, creative, needle-in-haystack) |
| `local-model bench <model>` | Speed benchmark at 512 / 2K / 8K / 32K / 64K context |
| `local-model add <path\|hf:repo> [name]` | Register a GGUF model (local file or Hugging Face) |
| `local-model info <model>` | Show model details, file size, GGUF metadata |
| `local-model config` | Show configuration (home dir, backends, platform) |

## Configuration

All state lives in `~/.local-model/` (override with `LOCAL_MODEL_HOME` env var):

```
~/.local-model/
├── config.json      # Backend paths and defaults
├── registry.json    # Registered models
├── models/          # GGUF files (downloaded or symlinked)
└── logs/            # Server logs, PID files, benchmark results
```

### Backends

A "backend" is a named path to a llama-server binary. Configure multiple backends for different llama.cpp forks:

```bash
local-model config --set-backend default /usr/local/bin/llama-server
local-model config --set-backend prismml /opt/prismml/build/bin/llama-server
local-model config --set-backend tqplus /opt/turboquant-plus/build/bin/llama-server
```

Then reference them in `registry.json` per-model:

```json
{
  "bonsai": { "binary": "prismml", ... },
  "gemma4": { "binary": "tqplus", ... }
}
```

If no backend is configured, `local-model` looks for `llama-server` on your `PATH`.

### Per-model options

| Field | Default | Description |
|-------|---------|-------------|
| `file` | — | GGUF filename (looked up in models dir) |
| `binary` | `"default"` | Backend name or absolute path to binary |
| `port` | `8080` | Server port |
| `context` | `8192` | Context window size |
| `cache_k` | `"f16"` | KV cache key type (`f16`, `q8_0`, `q4_0`) |
| `cache_v` | `"f16"` | KV cache value type (`f16`, `q8_0`, `turbo4`) |
| `flash_attn` | `"on"` | Flash attention (`on` / `off`) |
| `gpu_layers` | `99` | GPU layers to offload (`0` for CPU-only) |
| `threads` | `4` | CPU threads |
| `mmproj` | — | Vision multimodal projector GGUF file |
| `server_args` | `[]` | Extra args passed to llama-server |

## CPU-Only Setup (no GPU)

For machines without a discrete GPU (e.g. Intel UHD integrated graphics), set `gpu_layers` to `0`:

```json
{
  "bonsai": {
    "file": "Ternary-Bonsai-8B-Q2_0.gguf",
    "gpu_layers": 0,
    "threads": 8
  }
}
```

Ternary Bonsai 8B is an excellent choice for CPU-only inference — the 1.58-bit ternary weights are compute-friendly and the 2.2GB model fits easily in RAM.

## License

MIT
