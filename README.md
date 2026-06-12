# local-model-cli

Git-controlled home for Adam/OpenClaw's `local-model` CLI.

This repository tracks the implementation deployed at:

```text
~/.openclaw/workspace/scripts/local-model
```

The CLI manages local LLM inference servers in the OpenClaw workspace: model registry entries, launch commands, PID/log files, health checks, quality smoke tests, and speed benchmarks.

## Local deployment

```bash
git clone https://github.com/shakclaw-bot/local-model-cli.git ~/.openclaw/workspace/projects/local-model-cli
ln -sf ~/.openclaw/workspace/projects/local-model-cli/local_model/cli.py ~/.openclaw/workspace/scripts/local-model
chmod +x ~/.openclaw/workspace/projects/local-model-cli/local_model/cli.py
```

The script expects the OpenClaw workspace layout:

```text
~/.openclaw/workspace/
├── models/                 # GGUF/MLX models and registry.json
├── logs/                   # PID files, server logs, benchmark/test results
├── scripts/                # local-model symlink and backend launch scripts
└── tmp/                    # llama.cpp / MLX / Falcon runtime builds
```

## Commands

| Command | Description |
|---|---|
| `local-model list` | Show available models, ports, context, benchmark speeds, and status |
| `local-model start <model> [--ctx N]` | Start a model server and wait for `/health` |
| `local-model stop <model\|all>` | Stop one model or all registered models |
| `local-model status` | Show running model PIDs, health, logs, and slot info |
| `local-model test <model> [--prompts N]` | Run quality prompts and save `logs/test-<model>.json` |
| `local-model bench <model> [--ctx N]` | Run context/throughput benchmarks and save `logs/bench-<model>.json` |
| `local-model add <path\|hf:repo> [name]` | Register a new GGUF model |
| `local-model info <model>` | Show model details, paths, size, and GGUF header metadata |
| `local-model help` | Show command help plus registered models |

## Registry

Built-in defaults live in `local_model/cli.py` under `DEFAULT_REGISTRY`.

Custom/local additions live outside this repo at:

```text
~/.openclaw/workspace/models/registry.json
```

`load_registry()` merges `DEFAULT_REGISTRY` with that JSON file, so machine-specific entries such as experimental local builds do not need to be committed here.

## Runtime backends

The current implementation knows about these backend keys:

- `turboquant-plus`
- `turboquant`
- `prismml`
- `upstream`
- `mlx-vlm`
- `falcon`

Backend paths are resolved relative to `~/.openclaw/workspace/tmp/` in `BINARIES`.

## Current built-in model profiles

- `gemma4` — Gemma 4 E4B Q4_K_M, TurboQuant+, 128K context, port `8420`
- `gemma4-q8` — Gemma 4 E4B Q8_0, TurboQuant+, 32K context, port `8423`
- `bonsai` — Ternary Bonsai 8B Q2_0, PrismML, 64K context, port `8421`
- `bonsai-1bit` — legacy Bonsai 8B Q1_0, PrismML, 8K context, port `8427`
- `nemotron` — Nemotron 3 Nano 4B resolved from Ollama, TurboQuant+, 256K context, port `8422`
- `gemma4-mlx` — Gemma 4 E4B MLX 4-bit, port `8425`
- `gemma4-mlx-tq` — Gemma 4 E4B MLX + TurboQuant KV, port `8426`
- `gemma4-vision` — Falcon vision pipeline plus `gemma4`, port `8430`

## Verification

From the OpenClaw workspace:

```bash
python3 -m py_compile projects/local-model-cli/local_model/cli.py
scripts/local-model list
scripts/local-model info gemma4
```
