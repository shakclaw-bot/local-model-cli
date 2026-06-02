# local-model

A CLI tool for managing local LLM inference servers. Register GGUF models, start/stop servers, edit their settings, run speed benchmarks and accuracy evals — all from one command.

Works with any llama.cpp-compatible server binary (upstream llama.cpp, [PrismML](https://github.com/nicebread-cloud/prism-ml), [TurboQuant+](https://github.com/TheTom/turboquant_plus), etc.) on **macOS, Linux, and Windows**.

Highlights:
- **Cross-platform process management** — correct liveness checks and termination on Windows (no more accidentally killing your server on a status check) as well as POSIX.
- **VRAM-aware `-ncmoe` auto-sizing** for MoE models — picks how many expert layers to keep on the GPU based on free VRAM at launch.
- **Modern benchmarking** — streaming TTFT, decode tok/s percentiles, cold-prefill measurement.
- **Accuracy evals** — GSM8K reasoning (auto-scored, pulled live from Hugging Face) plus needle-in-haystack retrieval.
- **`edit` command** — change a model's name, port, context, runtime args, or key without hand-editing JSON.

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

Use `local-model edit` instead of hand-editing JSON:

```bash
local-model edit bonsai --context 65536 --threads 8 \
  --description "PrismML Q2_0 ternary. 2.18GB for 8B params."

# inspect the full config any time (no flags = inspector mode)
local-model edit bonsai
```

This writes to `~/.local-model/registry.json`, which you can still edit by hand if you prefer:

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

### 6. Benchmark and evaluate

```bash
local-model bench bonsai   # Speed: TTFT + decode tok/s (p50/p90) at a few context lengths
local-model eval bonsai    # Accuracy: GSM8K reasoning + needle retrieval (auto-scored)
local-model test bonsai    # Quick quality smoke test (reasoning, coding, factual, creative)
```

Example `bench` output:

```
 context   prompt   TTFT p50   TTFT p90   decode p50   decode p90   prefill
     512      434      0.45s      0.47s         48.7         49.0      1094
    8192     7187      4.96s      4.99s         44.1         44.7      1482
   32768    29155     19.84s     19.84s         35.9         36.0      1484
```

## Commands

| Command | Description |
|---------|-------------|
| `local-model list` | Show registered models with status and bench speeds |
| `local-model start <model> [--ctx N]` | Start a model server |
| `local-model serve <model> [--lan]` | Expose a model over Tailscale HTTPS and/or your LAN |
| `local-model stop <model\|all>` | Stop running server(s) |
| `local-model status` | Show running servers with health and slot info |
| `local-model scan [--register]` | Scan localhost and the LAN for OpenAI-compatible `/v1/models` endpoints |
| `local-model edit <model> [flags]` | Edit settings (name, port, context, runtime args, `--rename-key`); no flags = inspector |
| `local-model bench <model> [--ctx N] [--iters N]` | Speed benchmark: streaming TTFT + decode tok/s (p50/p90) |
| `local-model eval <model> [--questions N]` | Accuracy eval: GSM8K reasoning + needle retrieval (auto-scored) |
| `local-model test <model> [--prompts N]` | Quick quality smoke test (reasoning, coding, factual, creative) |
| `local-model add <path\|hf:repo> [name]` | Register a GGUF model (local file or Hugging Face) |
| `local-model info <model>` | Show model details, file size, GGUF metadata |
| `local-model completion install powershell` | Enable PowerShell tab completion |
| `local-model config [--set-backend N PATH]` | Show / edit configuration (home dir, backends, platform) |

## Shell completion

PowerShell tab completion can complete commands and registered model keys:

```powershell
local-model completion install powershell
```

This appends a small loader to your PowerShell profile. Open a new PowerShell session, then typing `local-model start qw` and pressing Tab completes matching registered models such as `qwen3.6-35b`.

For the current terminal only, run:

```powershell
local-model completion powershell | Out-String | Invoke-Expression
```

## Configuration

All state lives in `~/.local-model/` (override with `LOCAL_MODEL_HOME` env var):

```
~/.local-model/
├── config.json      # Backend paths and defaults
├── registry.json    # Registered models
├── models/          # GGUF files (downloaded, symlinked, or referenced by absolute path)
├── datasets/        # Cached eval datasets (e.g. GSM8K)
└── logs/            # Server logs, PID files, benchmark/eval results
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
| `flash_attn` | `"on"` | Flash attention (`on` / `off` / `auto`) |
| `gpu_layers` | `99` | GPU layers to offload (`0` for CPU-only) |
| `threads` | `4` | CPU threads |
| `mmproj` | — | Vision multimodal projector GGUF file |
| `server_args` | `[]` | Extra args passed to llama-server (e.g. `--no-mmap`, `--jinja`, `--chat-template-file`) |
| `auto_ncmoe` | — | VRAM-aware `-ncmoe` auto-sizing for MoE models (object — see below) |
| `notes` | — | Free-text description (shown by `info` / `edit`) |

## Editing models

`local-model edit` changes registry settings without hand-editing JSON:

```bash
# common fields
local-model edit bonsai --name "Ternary Bonsai 8B" --description "..." \
  --port 8090 --context 65536 --threads 8 --flash-attn on

# runtime / KV cache
local-model edit gemma4 --cache-k turbo3 --cache-v turbo3 --gpu-layers 99 \
  --server-args "--no-mmap --jinja --chat-template-file /path/to/template.jinja"

# arbitrary field with auto-typed value (int / float / bool / null / JSON)
local-model edit gemma4 --set threads=12
local-model edit gemma4 --set auto_ncmoe='{"safety_margin_mb":1500}'

# rename the key used in commands (moves the registry entry + pid/log/result files)
local-model edit qwen --rename-key qwen3.6-35b

# no flags = inspector: prints the full current config
local-model edit gemma4
```

`--server-args` replaces the existing list and is parsed with shell-style quoting. Editing a running model warns you that changes take effect on next start.

## Network discovery

`local-model scan` probes common model-server ports on localhost and your current `/24` LAN for OpenAI-compatible `/v1/models` endpoints:

```bash
local-model scan
local-model scan --target 100.64.0.0/24 --ports 8080,8000,11434
local-model scan --target 192.168.1.25:8080
local-model scan --register
```

When models are found, an interactive terminal shows a selectable list:

```text
Discovered models (Up/Down, Space to select, Enter to add, q to cancel):
  > [ ]  1. qwen3.6-35b  @  http://192.168.1.25:8080/v1
    [>] Add selected models
```

Use Up/Down to move, Space to tick models, then Enter on `Add selected models` to add them to `~/.local-model/registry.json`. Use `--register` to add every discovered model without prompting. Remote models appear in `local-model list` as `online`, `offline`, or `connected`.

To make a local model discoverable from another machine on your LAN, serve it with `--lan`:

```bash
# On the desktop running inference:
local-model serve bonsai --lan
# prints, for example: http://192.168.1.25:8080/v1

# On the laptop:
local-model scan --target 192.168.1.25:8080
local-model start qwen3.6-35b
```

`local-model start` on a remote model connects to the endpoint and prints the `base_url`, `model`, and `api_key` values to point a Pi agent or any OpenAI-compatible client at the desktop. `--lan` binds the model server to `0.0.0.0` and stays LAN-only; add `--tailscale` if you want both LAN and Tailscale exposure. Allow inbound TCP for the model port in your OS firewall if another device cannot reach it.

On Windows, `local-model scan` checks inbound TCP allow rules when you scan this machine's localhost/LAN IP and prints a `New-NetFirewallRule` command if the scanned port does not appear to have an allow rule.

## VRAM auto-sizing for MoE models (`auto_ncmoe`)

For Mixture-of-Experts models you can offload expert layers to CPU with llama.cpp's `-ncmoe`. Rather than hard-coding a value, add an `auto_ncmoe` block and the CLI queries free VRAM via `nvidia-smi` at start time and computes how many expert layers fit on the GPU:

```json
{
  "qwen3.6-35b": {
    "file": "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    "context": 262144,
    "cache_k": "turbo3",
    "cache_v": "turbo3",
    "auto_ncmoe": {
      "total_layers": 40,
      "per_layer_expert_mb": 310,
      "base_gpu_mb": 900,
      "compute_buffer_mb": 800,
      "rs_buffer_mb": 63,
      "kv_mb_at_128k": 500,
      "safety_margin_mb": 1024
    },
    "server_args": ["--no-mmap", "--jinja", "--chat-template-file", "/path/chat_template.jinja"]
  }
}
```

At start you'll see, e.g.:

```
  auto-ncmoe: 9770 MiB free, ctx=262144 -> 19/40 expert layers on GPU, ncmoe=21
```

The constants are empirical — measure them from a model's startup log (`CPU_Mapped`/`CUDA0 model buffer size`, `KV buffer size`, `RS buffer size`, `compute buffer size`). If `nvidia-smi` is unavailable, auto-sizing is skipped and you can set a static `-ncmoe` in `server_args` instead.

## Benchmarking & evaluation

**`bench`** measures speed with modern metrics: client-side **TTFT** (time to first token) via streaming, **decode tok/s** reported as p50/p90 across iterations (1 warmup discarded), and prefill tok/s. Each measured request uses a unique prefix to bust prompt-prefix caching, so TTFT/prefill reflect a true cold prefill.

```bash
local-model bench qwen3.6-35b --iters 3      # default 3 iters/length
```

**`eval`** measures accuracy:
- **GSM8K** grade-school math reasoning, pulled live from the Hugging Face datasets-server (no `datasets` dependency) and cached under `~/.local-model/datasets/`. Answers are auto-scored by exact numeric match.
- **Needle-in-haystack** retrieval at a couple of context lengths, auto-scored PASS/FAIL.

```bash
local-model eval qwen3.6-35b --questions 20   # default 20 GSM8K questions
```

Results are written to `~/.local-model/logs/bench-<key>.json` and `eval-<key>.json`, and the average speeds surface in `local-model list`.

## Platform support

Tested on macOS, Linux, and Windows. Windows specifics handled by the CLI:
- Liveness checks use `OpenProcess`/`GetExitCodeProcess` (a naive `os.kill(pid, 0)` would *terminate* the process on Windows).
- Process termination uses `taskkill /T /F` (reaps child processes); POSIX uses `SIGTERM`.
- `add` falls back to registering an absolute path when symlink creation isn't permitted (Windows without admin/Developer Mode).

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
