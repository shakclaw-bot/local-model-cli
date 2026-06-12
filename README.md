# local-model

A CLI tool for managing local LLM inference servers. Register GGUF models, start and stop servers, edit settings, run speed benchmarks, run quality checks, and discover compatible endpoints on your network from one command.

Use it to run and inspect local OpenAI-compatible model servers across **macOS, Linux, and Windows**. It works with any llama.cpp-compatible server binary, including upstream llama.cpp and compatible forks.

Highlights:
- **Cross-platform process management** - reliable liveness checks and termination on Windows, macOS, and Linux.
- **Simple model registry** - register local GGUF files or Hugging Face repositories and refer to them by a short key.
- **Backend configuration** - point each model at a named `llama-server` binary or use one binary for everything.
- **LAN discovery and sharing** - scan for OpenAI-compatible `/v1/models` endpoints and serve local models on your network.
- **Benchmarking and quality checks** - measure TTFT, decode tok/s, prefill speed, and run quick model smoke tests.
- **`edit` command** - change a model's name, port, context, runtime args, or key without hand-editing JSON.

## Install

```bash
# From GitHub
pip install git+https://github.com/shakclaw-bot/local-model-cli.git

# Reinstall or update from GitHub
pip install --upgrade --force-reinstall git+https://github.com/shakclaw-bot/local-model-cli.git

# Or clone and install locally
git clone https://github.com/shakclaw-bot/local-model-cli.git
cd local-model-cli
pip install -e .
```

Requirements:
- Python 3.10 or newer.
- A `llama-server` binary from llama.cpp or a compatible fork.
- A GGUF model file, or access to a Hugging Face repository containing one.

## Quick Start

### 1. Install or build a backend

`local-model` launches model servers through a `llama-server` binary. Install or build llama.cpp using the process approved for your environment, then note the path to the resulting binary.

Common examples:

```bash
# macOS / Linux example
local-model config --set-backend default /usr/local/bin/llama-server

# Windows example
local-model config --set-backend default C:\Tools\llama.cpp\build\bin\Release\llama-server.exe
```

If no backend is configured, `local-model` looks for `llama-server` on your `PATH`.

### 2. Register a model

Register a local GGUF file:

```bash
local-model add C:\Models\example-model.gguf work-model
```

Or register a model from Hugging Face:

```bash
local-model add hf:organization/model-repository work-model
```

The second argument is the local key used in commands. Pick short, stable names, such as `helpdesk`, `coding`, or `small-cpu`.

### 3. Review or edit settings

Use `local-model edit` instead of hand-editing JSON:

```bash
local-model edit work-model --name "Work Model" --port 8080 --context 8192 --threads 8

# Inspector mode: print the current model config
local-model edit work-model
```

This writes to `~/.local-model/registry.json`, which can still be edited by hand if needed:

```json
{
  "work-model": {
    "name": "Work Model",
    "file": "example-model.gguf",
    "binary": "default",
    "port": 8080,
    "context": 8192,
    "cache_k": "f16",
    "cache_v": "f16",
    "flash_attn": "on",
    "gpu_layers": 99,
    "threads": 8,
    "notes": "Internal model profile."
  }
}
```

### 4. Start and use the model

```bash
local-model start work-model
# Starting Work Model...
#   port: 8080  ctx: 8192  KV: f16/f16
#   waiting for health... ready
```

The server listens at `http://127.0.0.1:8080` by default and is compatible with OpenAI-format clients:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"work-model","messages":[{"role":"user","content":"Hello!"}]}'
```

### 5. Benchmark and test

```bash
local-model bench work-model   # Speed: TTFT + decode tok/s at several context lengths
local-model eval work-model    # Accuracy: GSM8K reasoning + needle retrieval
local-model test work-model    # Quick quality smoke test
```

Example `bench` output:

```text
 context   prompt   TTFT p50   TTFT p90   decode p50   decode p90   prefill
     512      434      0.45s      0.47s         48.7         49.0      1094
    8192     7187      4.96s      4.99s         44.1         44.7      1482
```

## Commands

| Command | Description |
|---------|-------------|
| `local-model list` | Show registered models with status and benchmark speeds |
| `local-model start <model> [--ctx N]` | Start a model server |
| `local-model serve <model> [--lan]` | Expose a model over Tailscale HTTPS and/or your LAN |
| `local-model stop <model\|all>` | Stop running server(s) |
| `local-model status` | Show running servers with health and slot info |
| `local-model scan [--register]` | Scan localhost and the LAN for OpenAI-compatible `/v1/models` endpoints |
| `local-model edit <model> [flags]` | Edit settings; no flags opens inspector mode |
| `local-model bench <model> [--ctx N] [--iters N]` | Speed benchmark: streaming TTFT + decode tok/s |
| `local-model eval <model> [--questions N]` | Accuracy eval: GSM8K reasoning + needle retrieval |
| `local-model test <model> [--prompts N]` | Quick quality smoke test |
| `local-model add <path\|hf:repo> [name]` | Register a GGUF model from a local file or Hugging Face |
| `local-model info <model>` | Show model details, file size, and GGUF metadata |
| `local-model completion install powershell` | Enable PowerShell tab completion |
| `local-model config [--set-backend N PATH]` | Show or edit configuration |

## Shell Completion

PowerShell tab completion can complete commands and registered model keys:

```powershell
local-model completion install powershell
```

This appends a small loader to your PowerShell profile. Open a new PowerShell session, then typing `local-model start wo` and pressing Tab completes matching registered models such as `work-model`.

For the current terminal only, run:

```powershell
local-model completion powershell | Out-String | Invoke-Expression
```

## Configuration

All state lives in `~/.local-model/` unless overridden with the `LOCAL_MODEL_HOME` environment variable:

```text
~/.local-model/
|-- config.json      # Backend paths and defaults
|-- registry.json    # Registered models
|-- models/          # GGUF files
|-- datasets/        # Cached eval datasets
`-- logs/            # Server logs, PID files, benchmark/eval results
```

### Backends

A backend is a named path to a `llama-server` binary. Configure multiple backends when different model formats or hardware profiles need different llama.cpp builds:

```bash
local-model config --set-backend default /usr/local/bin/llama-server
local-model config --set-backend cpu /opt/llama.cpp-cpu/build/bin/llama-server
local-model config --set-backend gpu /opt/llama.cpp-cuda/build/bin/llama-server
```

Then reference them per model:

```json
{
  "small-cpu": { "binary": "cpu" },
  "work-model": { "binary": "gpu" }
}
```

### Per-model Options

| Field | Default | Description |
|-------|---------|-------------|
| `file` | - | GGUF filename or absolute path |
| `binary` | `"default"` | Backend name or absolute path to binary |
| `port` | `8080` | Server port |
| `context` | `8192` | Context window size |
| `cache_k` | `"f16"` | KV cache key type, such as `f16`, `q8_0`, or `q4_0` |
| `cache_v` | `"f16"` | KV cache value type, such as `f16`, `q8_0`, or backend-specific values |
| `flash_attn` | `"on"` | Flash attention: `on`, `off`, or `auto` |
| `gpu_layers` | `99` | GPU layers to offload; use `0` for CPU-only |
| `threads` | `4` | CPU threads |
| `mmproj` | - | Vision multimodal projector GGUF file |
| `server_args` | `[]` | Extra args passed to `llama-server` |
| `auto_ncmoe` | - | VRAM-aware `-ncmoe` auto-sizing for MoE models |
| `notes` | - | Free-text description shown by `info` and `edit` |

## Editing Models

`local-model edit` changes registry settings without hand-editing JSON:

```bash
# Common fields
local-model edit work-model --name "Work Model" --description "Internal model profile" \
  --port 8090 --context 8192 --threads 8 --flash-attn on

# Runtime and KV cache options
local-model edit work-model --cache-k f16 --cache-v f16 --gpu-layers 99 \
  --server-args "--no-mmap --jinja --chat-template-file /path/to/template.jinja"

# Arbitrary field with auto-typed value: int, float, bool, null, or JSON
local-model edit work-model --set threads=12
local-model edit work-model --set auto_ncmoe='{"safety_margin_mb":1500}'

# Rename the key used in commands
local-model edit work-model --rename-key team-model

# Inspector mode
local-model edit team-model
```

`--server-args` replaces the existing list and is parsed with shell-style quoting. Editing a running model warns you that changes take effect on next start.

## Network Discovery

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
  > [ ]  1. work-model  @  http://192.168.1.25:8080/v1
    [>] Add selected models
```

Use Up/Down to move, Space to tick models, then Enter on `Add selected models` to add them to `~/.local-model/registry.json`. Use `--register` to add every discovered model without prompting. Remote models appear in `local-model list` as `online`, `offline`, or `connected`.

To make a local model discoverable from another machine on your LAN, serve it with `--lan`:

```bash
# On the machine running inference:
local-model serve work-model --lan
# prints, for example: http://192.168.1.25:8080/v1

# On another machine:
local-model scan --target 192.168.1.25:8080
local-model start work-model
```

`local-model start` on a remote model connects to the endpoint and prints the `base_url`, `model`, and `api_key` values for any OpenAI-compatible client. `--lan` binds the model server to `0.0.0.0` and stays LAN-only; add `--tailscale` if you want both LAN and Tailscale exposure. Allow inbound TCP for the model port in your OS firewall if another device cannot reach it.

On Windows, `local-model scan` checks inbound TCP allow rules when you scan this machine's localhost/LAN IP and prints a `New-NetFirewallRule` command if the scanned port does not appear to have an allow rule.

## VRAM Auto-sizing for MoE Models

For Mixture-of-Experts models, llama.cpp can offload expert layers to CPU with `-ncmoe`. Instead of hard-coding a value, add an `auto_ncmoe` block and the CLI queries free VRAM via `nvidia-smi` at start time to compute how many expert layers fit on the GPU:

```json
{
  "work-moe": {
    "file": "example-moe-model.gguf",
    "context": 32768,
    "cache_k": "f16",
    "cache_v": "f16",
    "auto_ncmoe": {
      "total_layers": 40,
      "per_layer_expert_mb": 310,
      "base_gpu_mb": 900,
      "compute_buffer_mb": 800,
      "rs_buffer_mb": 63,
      "kv_mb_at_128k": 500,
      "safety_margin_mb": 1024
    },
    "server_args": ["--no-mmap"]
  }
}
```

At start you will see output similar to:

```text
  auto-ncmoe: 9770 MiB free, ctx=32768 -> 19/40 expert layers on GPU, ncmoe=21
```

The constants are empirical. Measure them from a model's startup log, including `CPU_Mapped` or `CUDA0 model buffer size`, `KV buffer size`, `RS buffer size`, and `compute buffer size`. If `nvidia-smi` is unavailable, auto-sizing is skipped and you can set a static `-ncmoe` in `server_args` instead.

## Benchmarking and Evaluation

`bench` measures speed with modern metrics: client-side TTFT (time to first token) via streaming, decode tok/s reported as p50/p90 across iterations, and prefill tok/s. Each measured request uses a unique prefix to avoid prompt-prefix cache effects.

```bash
local-model bench work-model --iters 3
```

`eval` measures accuracy:
- **GSM8K** grade-school math reasoning, pulled from the Hugging Face datasets-server and cached under `~/.local-model/datasets/`.
- **Needle-in-haystack** retrieval at a few context lengths.

```bash
local-model eval work-model --questions 20
```

Results are written to `~/.local-model/logs/bench-<key>.json` and `eval-<key>.json`, and average speeds appear in `local-model list`.

## CPU-only Setup

For machines without a discrete GPU, set `gpu_layers` to `0`:

```bash
local-model edit work-model --gpu-layers 0 --threads 8
```

You can also set it directly in the registry:

```json
{
  "work-model": {
    "file": "example-model.gguf",
    "gpu_layers": 0,
    "threads": 8
  }
}
```

CPU-only performance depends heavily on the model size, quantization, memory bandwidth, and thread count. Start with a smaller GGUF model for general workstation use, then benchmark before distributing a standard profile.

## Deployment Notes

- Decide where approved GGUF files should live, such as `C:\Models` on Windows or `/opt/models` on Linux.
- Decide whether the backend binary is installed on `PATH` or configured with `local-model config --set-backend`.
- Use stable model keys so scripts and client configuration do not need to change.
- Keep LAN serving limited to trusted networks and review firewall rules before enabling access from other machines.
- Use `LOCAL_MODEL_HOME` if IT needs the registry, logs, and model cache under a managed path.

## Platform Support

Tested on macOS, Linux, and Windows. Windows-specific behavior handled by the CLI:
- Liveness checks avoid Windows signal behavior that can terminate a process during status checks.
- Process termination uses `taskkill /T /F` to reap child processes.
- `add` falls back to registering an absolute path when symlink creation is not permitted.
- `scan` can suggest a PowerShell firewall command when an inbound TCP allow rule appears to be missing.

## License

MIT
