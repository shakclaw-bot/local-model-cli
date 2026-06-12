#!/usr/bin/env python3
"""local-model — manage local LLM inference servers.

Usage:
  local-model list                        Show available models and their status
  local-model start <model> [--ctx N]     Start a model server
  local-model stop <model|all>            Stop a running model server
  local-model status                      Show running servers with health + memory
  local-model test <model> [--prompts N]  Run quality tests against a running model
  local-model bench <model> [--ctx N]     Run speed benchmark
  local-model add <path|hf-repo> [name]   Register a new GGUF model
  local-model info <model>                Show model details (arch, params, quant, ctx)
"""
from __future__ import annotations
import argparse, json, os, re, signal, subprocess, sys, textwrap, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path.home() / ".openclaw/workspace"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
SCRIPTS_DIR = ROOT / "scripts"
REGISTRY_FILE = ROOT / "models/registry.json"

# ── Model Registry ──────────────────────────────────────────────────────────

DEFAULT_REGISTRY = {
    "gemma4": {
        "name": "Gemma 4 E4B",
        "file": "gemma-4-E4B-it-Q4_K_M.gguf",
        "binary": "turboquant-plus",
        "port": 8420,
        "context": 131072,
        "cache_k": "f16",
        "cache_v": "f16",
        "flash_attn": "on",
        "threads": 4,
        "notes": "Best with TQ+ f16 KV. 128K native context. Supports thinking mode and vision.",
        "mmproj": "mmproj-gemma4-e4b-F16.gguf",
    },
    "gemma4-q8": {
        "name": "Gemma 4 E4B (Q8_0)",
        "file": "gemma-4-E4B-it-Q8_0.gguf",
        "binary": "turboquant-plus",
        "port": 8423,
        "context": 32768,
        "cache_k": "f16",
        "cache_v": "f16",
        "flash_attn": "on",
        "threads": 4,
        "notes": "Higher quality Q8_0 quant. 7.6GB — use for quality-critical tasks at shorter context. 32K safe max on 16GB. Vision supported.",
        "mmproj": "mmproj-gemma4-e4b-F16.gguf",
    },
    "bonsai": {
        "name": "Ternary Bonsai 8B (1.58-bit)",
        "file": "Ternary-Bonsai-8B-Q2_0.gguf",
        "binary": "prismml",
        "port": 8421,
        "context": 65536,
        "cache_k": "f16",
        "cache_v": "f16",
        "flash_attn": "on",
        "threads": 4,
        "notes": "PrismML Q2_0 ternary. 2.18GB for 8B params. Based on Qwen3-8B. 65K native context.",
    },
    "bonsai-1bit": {
        "name": "Bonsai 8B (1-bit, legacy)",
        "file": "Bonsai-8B.gguf",
        "binary": "prismml",
        "port": 8427,
        "context": 8192,
        "cache_k": "f16",
        "cache_v": "f16",
        "flash_attn": "off",
        "threads": 4,
        "notes": "PrismML Q1_0 format. 1.15GB for 8B params. Max 8K context. Legacy — use 'bonsai' for ternary.",
    },
    "nemotron": {
        "name": "Nemotron 3 Nano 4B",
        "source": "ollama:nemotron-3-nano:4b",
        "binary": "turboquant-plus",
        "port": 8422,
        "context": 262144,
        "cache_k": "q8_0",
        "cache_v": "turbo4",
        "flash_attn": "on",
        "threads": 4,
        "notes": "Resolved from Ollama. 256K native context. turbo4 KV for memory savings.",
    },
    "gemma4-mlx": {
        "name": "Gemma 4 E4B (MLX 4-bit)",
        "dir": "gemma-4-e4b-it-4bit-mlx",
        "binary": "mlx-vlm",
        "port": 8425,
        "context": 65536,
        "notes": "MLX native inference. f16 KV cache. 64K safe max on 16GB.",
    },
    "gemma4-mlx-tq": {
        "name": "Gemma 4 E4B (MLX TurboQuant)",
        "dir": "gemma-4-e4b-it-4bit-mlx",
        "binary": "mlx-vlm",
        "port": 8426,
        "context": 81920,
        "kv_bits": 3.5,
        "kv_quant_scheme": "turboquant",
        "notes": "MLX + TurboQuant 3.5-bit KV. 80K safe max on 16GB M4.",
    },
    "gemma4-vision": {
        "name": "Gemma 4 Vision Pipeline",
        "binary": "falcon",
        "port": 8430,
        "context": 131072,
        "notes": "Falcon OCR (0.3B) + Falcon Perception (0.6B) + Gemma 4 TQ+ Q4_K_M (llama.cpp). 128K context, ~6.5 GB total RAM.",
        "requires": "gemma4",
        "script": "start-gemma4-vision.sh",
    },
}

BINARIES = {
    "turboquant-plus": ROOT / "tmp/llama-cpp-turboquant-plus/build/bin/llama-server",
    "turboquant": ROOT / "tmp/llama-cpp-turboquant/build/bin/llama-server",
    "prismml": ROOT / "tmp/llama-cpp-prismml/build/bin/llama-server",
    "upstream": ROOT / "tmp/llama-cpp-upstream/build/bin/llama-server",
    "mlx-vlm": ROOT / "tmp/mlx-vlm-env/bin/python3",
    "falcon": ROOT / "tmp/falcon-env/bin/python3",
}


def is_mlx_backend(model_cfg):
    return model_cfg.get("binary") in ("mlx-vlm", "falcon")


def is_script_backend(model_cfg):
    return model_cfg.get("script") is not None


def load_registry():
    if REGISTRY_FILE.exists():
        try:
            custom = json.loads(REGISTRY_FILE.read_text())
            merged = {**DEFAULT_REGISTRY, **custom}
            return merged
        except Exception:
            pass
    return dict(DEFAULT_REGISTRY)


def save_registry(registry):
    # Only save entries not in defaults (custom models)
    custom = {k: v for k, v in registry.items() if k not in DEFAULT_REGISTRY}
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(custom, indent=2) + "\n")


def get_model(registry, name):
    if name in registry:
        return registry[name]
    # Fuzzy match
    matches = [k for k in registry if name.lower() in k.lower()]
    if len(matches) == 1:
        return registry[matches[0]]
    if len(matches) > 1:
        print(f"Ambiguous model '{name}'. Matches: {', '.join(matches)}", file=sys.stderr)
        sys.exit(1)
    print(f"Unknown model '{name}'. Run 'local-model list' to see available models.", file=sys.stderr)
    sys.exit(1)


def get_model_key(registry, name):
    if name in registry:
        return name
    matches = [k for k in registry if name.lower() in k.lower()]
    if len(matches) == 1:
        return matches[0]
    return name


# ── Binary / Path Resolution ────────────────────────────────────────────────

def resolve_binary(model_cfg):
    bin_key = model_cfg.get("binary", "turboquant-plus")
    if bin_key in BINARIES:
        return str(BINARIES[bin_key])
    return bin_key  # treat as direct path


def resolve_model_path(model_cfg):
    source = model_cfg.get("source", "")
    if source.startswith("ollama:"):
        model_id = source[len("ollama:"):]
        for bin_path in ["ollama", "/tmp/ollama-rc/ollama"]:
            try:
                out = subprocess.check_output(
                    [bin_path, "show", "--modelfile", model_id],
                    stderr=subprocess.DEVNULL, timeout=10
                ).decode()
                for line in out.splitlines():
                    if line.startswith("FROM "):
                        p = line.split(None, 1)[1].strip()
                        if os.path.isfile(p):
                            return p
            except Exception:
                continue
        return None

    # MLX models are directories
    model_dir = model_cfg.get("dir", "")
    if model_dir:
        p = MODELS_DIR / model_dir
        if p.is_dir():
            return str(p)
        if os.path.isdir(model_dir):
            return model_dir

    model_file = model_cfg.get("file", "")
    if model_file:
        p = MODELS_DIR / model_file
        if p.exists():
            return str(p)
        # Try as absolute path
        if os.path.isfile(model_file):
            return model_file

    # Script-based models don't need a model path
    if model_cfg.get("script"):
        return "(script)"

    return None


# ── PID Management ───────────────────────────────────────────────────────────

def pid_file_for(key):
    return LOGS_DIR / f"{key}.pid"


def log_file_for(key):
    return LOGS_DIR / f"{key}.log"


def get_running_pid(key):
    pf = pid_file_for(key)
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, 0)  # check alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pf.unlink(missing_ok=True)
        return None


def check_health(port, timeout=3):
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout)
        body = r.read()
        if not body:
            return 200 <= getattr(r, "status", 200) < 500
        status = json.loads(body).get("status", "")
        return status in ("ok", "healthy")
    except Exception:
        return False


# ── Commands ─────────────────────────────────────────────────────────────────

def _get_bench_speeds(key):
    """Load average prompt (in) and gen (out) tok/s from most recent bench or test results."""
    for filename in [f"bench-{key}.json", f"test-{key}.json"]:
        path = LOGS_DIR / filename
        if path.exists():
            try:
                data = json.loads(path.read_text())
                gen_speeds = [r.get("gen_tps", 0) for r in data if isinstance(r, dict) and r.get("gen_tps", 0) > 0]
                prompt_speeds = [r.get("prompt_tps", 0) for r in data if isinstance(r, dict) and r.get("prompt_tps", 0) > 0]
                if gen_speeds:
                    avg_gen = round(sum(gen_speeds) / len(gen_speeds), 1)
                    avg_prompt = round(sum(prompt_speeds) / len(prompt_speeds), 1) if prompt_speeds else None
                    return avg_prompt, avg_gen
            except Exception:
                pass
    return None, None


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(value) -> int:
    return len(_ANSI_RE.sub("", str(value)))


def _pad(value, width: int, align: str = "<") -> str:
    """Pad while ignoring ANSI colour sequences in the visible width."""
    text = str(value)
    pad = max(0, width - _visible_len(text))
    if align == ">":
        return " " * pad + text
    return text + " " * pad


def cmd_list(args):
    registry = load_registry()
    rows = []
    for key, cfg in sorted(registry.items()):
        port = cfg.get("port", "?")
        ctx = cfg.get("context", "?")
        if isinstance(ctx, int):
            ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        else:
            ctx_str = str(ctx)

        prompt_speed, gen_speed = _get_bench_speeds(key)
        in_str = f"{prompt_speed:.0f}" if prompt_speed else "\033[90m—\033[0m"
        out_str = f"{gen_speed:.1f}" if gen_speed else "\033[90m—\033[0m"

        pid = get_running_pid(key)
        if pid and check_health(port):
            status = f"\033[32mrunning\033[0m (:{port})"
        elif pid:
            status = f"\033[33mstarting\033[0m"
        else:
            status = "\033[90mstopped\033[0m"

        model_path = resolve_model_path(cfg)
        if not model_path and not cfg.get("source", "").startswith("ollama:"):
            status = "\033[31mmissing\033[0m"

        rows.append([
            key,
            cfg.get("name", "?"),
            str(port),
            ctx_str,
            in_str,
            out_str,
            status,
        ])

    headers = ["Model", "Name", "Port", "Context", "tok/s in", "tok/s out", "Status"]
    aligns = ["<", "<", ">", ">", ">", ">", "<"]
    widths = []
    for idx, header in enumerate(headers):
        values = [row[idx] for row in rows]
        widths.append(max(_visible_len(header), *( _visible_len(v) for v in values)) if values else _visible_len(header))

    def fmt(row):
        return "  ".join(_pad(value, widths[idx], aligns[idx]) for idx, value in enumerate(row))

    header_line = fmt(headers)
    print(header_line)
    print("-" * _visible_len(header_line))
    for row in rows:
        print(fmt(row))


def _build_server_cmd(cfg, binary, model_path, port, ctx):
    """Build the server launch command for either llama.cpp or mlx-vlm."""
    if is_mlx_backend(cfg):
        cmd = [
            binary, "-m", "mlx_vlm", "server",
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        if cfg.get("kv_bits") is not None:
            cmd += ["--kv-bits", str(cfg["kv_bits"])]
        if cfg.get("kv_quant_scheme"):
            cmd += ["--kv-quant-scheme", cfg["kv_quant_scheme"]]
        if ctx:
            cmd += ["--max-kv-size", str(ctx)]
        if cfg.get("draft_model"):
            cmd += ["--draft-model", str(cfg["draft_model"])]
        if cfg.get("draft_kind"):
            cmd += ["--draft-kind", str(cfg["draft_kind"])]
        if cfg.get("draft_block_size") is not None:
            cmd += ["--draft-block-size", str(cfg["draft_block_size"])]
        return cmd

    cmd = [
        binary, "-m", model_path,
        "-ngl", "99",
        "-c", str(ctx),
        "-fa", cfg.get("flash_attn", "on"),
        "-ctk", cfg.get("cache_k", "f16"),
        "-ctv", cfg.get("cache_v", "f16"),
        "--threads", str(cfg.get("threads", 4)),
        "-np", "1",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    mmproj = cfg.get("mmproj")
    if mmproj:
        mmproj_path = MODELS_DIR / mmproj
        if mmproj_path.exists():
            cmd += ["-mm", str(mmproj_path)]
        else:
            print(f"  warning: mmproj not found: {mmproj_path}", file=sys.stderr)
    return cmd


def _describe_config(cfg):
    """Return a short config description for startup messages."""
    if is_script_backend(cfg):
        return f"script: {cfg['script']}"
    if cfg.get("binary") == "mlx-vlm":
        kv = f"TurboQuant {cfg['kv_bits']}-bit" if cfg.get("kv_bits") else "f16 KV"
        if cfg.get("draft_model"):
            kv += f" + {cfg.get('draft_kind', 'draft')} draft"
        return f"mlx-vlm  {kv}"
    if cfg.get("binary") == "falcon":
        return "falcon-perception MLX"
    return f"KV: {cfg.get('cache_k', '?')}/{cfg.get('cache_v', '?')}"


def cmd_start(args):
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    # Already running?
    pid = get_running_pid(key)
    if pid and check_health(cfg["port"]):
        print(f"{cfg['name']} is already running on port {cfg['port']} (PID {pid})")
        return

    port = cfg.get("port", 8420)

    # Kill stale process on same port
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except ProcessLookupError:
            pass

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_f = log_file_for(key)

    # Script-based models (e.g. gemma4-vision) use a launch script
    if is_script_backend(cfg):
        script = SCRIPTS_DIR / cfg["script"]
        if not script.exists():
            print(f"Launch script not found: {script}", file=sys.stderr)
            sys.exit(1)

        # Start prerequisite model if needed
        requires = cfg.get("requires")
        if requires:
            req_cfg = registry.get(requires, {})
            req_port = req_cfg.get("port")
            req_pid = get_running_pid(requires)
            if not (req_pid and check_health(req_port)):
                print(f"Starting prerequisite: {req_cfg.get('name', requires)}...")
                # Recursively start the required model
                class FakeArgs:
                    def __init__(self, model):
                        self.model = model
                        self.ctx = None
                cmd_start(FakeArgs(requires))

        print(f"Starting {cfg['name']}...")
        print(f"  port: {port}  script: {script.name}")

        env = {**os.environ, "PORT": str(port)}
        with open(log_f, "w") as lf:
            proc = subprocess.Popen(
                ["bash", str(script)], stdout=lf, stderr=subprocess.STDOUT, env=env,
            )

        # The script handles its own PID file and health wait, but we track the script PID
        # Wait for health directly
        print("  waiting for health...", end="", flush=True)
        timeout = 180
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if check_health(port):
                elapsed = time.monotonic() - t0
                pid_f = LOGS_DIR / f"{key}.pid"
                if pid_f.exists():
                    print(f" ready ({elapsed:.0f}s)")
                    print(f"\n{cfg['name']} is running on port {port}.")
                else:
                    pid_file_for(key).write_text(str(proc.pid))
                    print(f" ready ({elapsed:.0f}s)")
                    print(f"\n{cfg['name']} is running on port {port}.")
                return
            time.sleep(1)
            print(".", end="", flush=True)

        print(f"\n  Timed out after {timeout}s. Check {log_f}")
        sys.exit(1)

    binary = resolve_binary(cfg)
    if not os.path.isfile(binary):
        print(f"Binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model_path(cfg)
    if not model_path:
        print(f"Model not found for {key}", file=sys.stderr)
        sys.exit(1)

    ctx = args.ctx or cfg.get("context", 8192)

    cmd = _build_server_cmd(cfg, binary, model_path, port, ctx)

    print(f"Starting {cfg['name']}...")
    print(f"  port: {port}  ctx: {ctx}  {_describe_config(cfg)}")

    with open(log_f, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

    pid_file_for(key).write_text(str(proc.pid))
    print(f"  pid: {proc.pid}  log: {log_f}")

    # Wait for health
    print("  waiting for health...", end="", flush=True)
    timeout = 180
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            print(f"\n  Server exited! Check {log_f}")
            sys.exit(1)
        if check_health(port):
            elapsed = time.monotonic() - t0
            print(f" ready ({elapsed:.0f}s)")
            print(f"\n{cfg['name']} is running on port {port}.")
            return
        time.sleep(1)
        print(".", end="", flush=True)

    print(f"\n  Timed out after {timeout}s. Check {log_f}")
    sys.exit(1)


def cmd_stop(args):
    registry = load_registry()

    if args.model == "all":
        targets = list(registry.keys())
    else:
        targets = [get_model_key(registry, args.model)]

    stopped = 0
    for key in targets:
        pid = get_running_pid(key)
        if pid:
            name = registry.get(key, {}).get("name", key)
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Stopped {name} (PID {pid})")
                stopped += 1
            except ProcessLookupError:
                pass
            pid_file_for(key).unlink(missing_ok=True)

    if stopped == 0:
        print("No running models to stop.")


def cmd_status(args):
    registry = load_registry()
    found = False

    for key, cfg in sorted(registry.items()):
        pid = get_running_pid(key)
        if not pid:
            continue
        found = True
        port = cfg.get("port", "?")
        healthy = check_health(port) if isinstance(port, int) else False

        print(f"\n{cfg.get('name', key)}")
        print(f"  PID:    {pid}")
        print(f"  Port:   {port}")
        print(f"  Health: {'OK' if healthy else 'NOT READY'}")
        print(f"  Log:    {log_file_for(key)}")

        # Try to get slot info
        if healthy:
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=3)
                slots = json.loads(r.read())
                for s in slots:
                    print(f"  Slot {s.get('id', '?')}: {s.get('n_decoded', 0)} tokens decoded, state={s.get('state', '?')}")
            except Exception:
                pass

    if not found:
        print("No models currently running.")


def cmd_test(args):
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    # Check for other running models
    _ensure_clean_for_bench(registry, key)

    # Start model if not running
    port, started_by_us = _start_for_bench(registry, key)
    model_name = resolve_model_path(cfg) if is_mlx_backend(cfg) else "test"

    tests = [
        ("Reasoning", "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain step by step.", 600),
        ("Coding", "Write a Python function that returns the longest increasing subsequence of a list of integers. Include a brief explanation.", 800),
        ("Factual", "What are the three laws of thermodynamics? One sentence each.", 512),
        ("Creative", "Write a short paragraph describing a city at night from the perspective of a cat on a rooftop.", 512),
        ("Summarize", (
            "Here is a technical document about database indexing:\n\n"
            "B-tree indexes are the most common type of database index. They maintain sorted data "
            "and allow searches, sequential access, insertions, and deletions in logarithmic time. "
            "PostgreSQL uses B-tree indexes by default. GIN indexes are preferred for full-text search "
            "and array containment queries. GiST indexes support complex data types like geometric shapes "
            "and ranges. BRIN indexes are efficient for very large tables where data is physically ordered. "
            "Partial indexes only index rows matching a predicate. Expression indexes allow indexing "
            "computed values. Covering indexes store additional columns to enable index-only scans.\n\n"
            "Summarize the key differences between these index types in a brief comparison."
        ), 600),
        ("Needle-2K", f"Read carefully:\n\n{_build_haystack(2000)}\n\nWhat is the secret project codename? Answer with just the codename.", 256),
    ]

    if args.prompts and args.prompts < len(tests):
        tests = tests[:args.prompts]

    print(f"Testing {cfg['name']} on port {port}\n")
    print(f"{'Test':<16} {'tok/s':>7} {'Tokens':>7} {'Time':>7}  Preview")
    print("-" * 80)

    results = []
    for label, prompt, max_tok in tests:
        try:
            r = _chat(port, prompt, max_tok, model_name=model_name)
            preview = r["content"].replace("\n", " ")[:50]
            print(f"{label:<16} {r['gen_tps']:>6.1f} {r['tokens']:>7} {r['elapsed']:>6.1f}s  {preview}...")
            results.append({"label": label, **r})
        except Exception as e:
            print(f"{label:<16} {'ERROR':>7}  {e}")

    if results:
        avg_tps = sum(r["gen_tps"] for r in results) / len(results)
        print(f"\nAverage generation: {avg_tps:.1f} tok/s")

    out = LOGS_DIR / f"test-{key}.json"
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"Results saved to {out}")

    if started_by_us:
        pid = get_running_pid(key)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            pid_file_for(key).unlink(missing_ok=True)
            print(f"\n{cfg['name']} stopped (was started for test).")


def _ensure_clean_for_bench(registry, target_key):
    """Check for other running models, prompt to stop them, then start the target."""
    others_running = []
    for k, c in registry.items():
        if k == target_key:
            continue
        pid = get_running_pid(k)
        if pid:
            others_running.append((k, c.get("name", k), pid))

    if others_running:
        print("Other models are currently running:")
        for k, name, pid in others_running:
            print(f"  - {name} (PID {pid})")
        print("\nBenchmarks should run in isolation for accurate results.")
        try:
            answer = input("Stop them before benchmarking? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "y"
        if answer in ("", "y", "yes"):
            for k, name, pid in others_running:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"  Stopped {name}")
                except ProcessLookupError:
                    pass
                pid_file_for(k).unlink(missing_ok=True)
            time.sleep(3)
        else:
            print("Continuing with other models running (results may be affected by RAM pressure).\n")


def _start_for_bench(registry, key, ctx_override=None):
    """Start a model server for benchmarking. Returns (port, started_by_us)."""
    cfg = registry[key]
    port = cfg.get("port", 8420)

    # Already running and healthy?
    pid = get_running_pid(key)
    if pid and check_health(port):
        return port, False

    # Need to start it
    binary = resolve_binary(cfg)
    if not os.path.isfile(binary):
        print(f"Binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model_path(cfg)
    if not model_path:
        print(f"Model file not found for {key}", file=sys.stderr)
        sys.exit(1)

    ctx = ctx_override or cfg.get("context", 8192)

    # Kill stale process
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except ProcessLookupError:
            pass

    cmd = _build_server_cmd(cfg, binary, model_path, port, ctx)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_f = log_file_for(key)

    print(f"Starting {cfg['name']} for benchmark...")
    print(f"  port: {port}  ctx: {ctx}  {_describe_config(cfg)}")

    with open(log_f, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

    pid_file_for(key).write_text(str(proc.pid))

    print("  waiting for health...", end="", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 180:
        if proc.poll() is not None:
            print(f"\n  Server exited! Check {log_f}")
            sys.exit(1)
        if check_health(port):
            print(f" ready ({time.monotonic() - t0:.0f}s)")
            return port, True
        time.sleep(1)
        print(".", end="", flush=True)

    print(f"\n  Timed out. Check {log_f}")
    sys.exit(1)


def cmd_bench(args):
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    # 1. Check for other running models
    _ensure_clean_for_bench(registry, key)

    # 2. Start the target model
    ctx_override = args.ctx or cfg.get("context", 8192)
    port, started_by_us = _start_for_bench(registry, key, ctx_override)

    # 3. Query actual context from server (llama.cpp only; mlx-vlm has no /slots)
    actual_ctx = None
    if not is_mlx_backend(cfg):
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=3)
            slots = json.loads(r.read())
            if slots:
                actual_ctx = slots[0].get("n_ctx")
        except Exception:
            pass
    ctx_limit = actual_ctx or ctx_override
    model_name = resolve_model_path(cfg) if is_mlx_backend(cfg) else "test"

    contexts = [512, 2048, 8192, 32768, 65536]
    contexts = [c for c in contexts if c <= ctx_limit]

    print(f"\nBenchmarking {cfg['name']} on port {port} (ctx={ctx_limit})\n")

    filler = "The history of computing is a fascinating journey from Babbage to quantum computers. Each generation built on the last with vacuum tubes giving way to transistors then integrated circuits. Software evolved from machine code to high-level languages. Networks connected computers globally. AI and ML represent the latest frontier. "

    results = []
    for target_tokens in contexts:
        fill_target = int(target_tokens * 0.6)
        n_repeats = max(1, fill_target // 80)
        text = "\n".join([f"[{i}] {filler}" for i in range(n_repeats)])
        text += "\nSummarize the above in 2 sentences."

        label = f"~{target_tokens} tok"
        try:
            r = _chat(port, text, max_tokens=128, model_name=model_name)
            print(f"  {label:<12} prompt={r['prompt_tokens']:>6} tok  prompt_speed={r['prompt_tps']:>6.1f} tok/s  gen={r['gen_tps']:>5.1f} tok/s  wall={r['elapsed']:.1f}s")
            results.append({"context": target_tokens, **r})
        except Exception as e:
            print(f"  {label:<12} ERROR: {e}")

    if results:
        print(f"\nPeak generation: {max(r['gen_tps'] for r in results):.1f} tok/s")
        print(f"Peak prompt:     {max(r['prompt_tps'] for r in results):.1f} tok/s")

    out = LOGS_DIR / f"bench-{key}.json"
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"Results saved to {out}")

    # 4. Stop the model if we started it
    if started_by_us:
        pid = get_running_pid(key)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            pid_file_for(key).unlink(missing_ok=True)
            print(f"\n{cfg['name']} stopped (was started for benchmark).")


def cmd_add(args):
    registry = load_registry()
    source = args.source
    name = args.name

    # Detect source type
    if source.startswith("hf:") or "/" in source and not os.path.exists(source):
        # Hugging Face download
        hf_repo = source.replace("hf:", "")
        if not name:
            name = hf_repo.split("/")[-1].lower().replace(" ", "-")

        print(f"Downloading from Hugging Face: {hf_repo}")
        print(f"Looking for GGUF files...")

        # List files in repo
        try:
            api_url = f"https://huggingface.co/api/models/{hf_repo}"
            r = urllib.request.urlopen(api_url, timeout=30)
            repo_info = json.loads(r.read())
            siblings = repo_info.get("siblings", [])
            gguf_files = [s["rfilename"] for s in siblings if s["rfilename"].endswith(".gguf")]
        except Exception as e:
            print(f"Failed to query HF API: {e}", file=sys.stderr)
            sys.exit(1)

        if not gguf_files:
            print(f"No GGUF files found in {hf_repo}", file=sys.stderr)
            sys.exit(1)

        print(f"\nAvailable GGUF files:")
        for i, f in enumerate(gguf_files, 1):
            print(f"  {i}) {f}")

        if len(gguf_files) == 1:
            choice = 0
        else:
            try:
                choice = int(input("Choose file number: ")) - 1
            except (ValueError, EOFError):
                print("Invalid selection", file=sys.stderr)
                sys.exit(1)

        gguf_name = gguf_files[choice]
        dest = MODELS_DIR / gguf_name
        dl_url = f"https://huggingface.co/{hf_repo}/resolve/main/{gguf_name}"

        print(f"\nDownloading {gguf_name}...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-L", "-o", str(dest), "--progress-bar", dl_url], check=True)
        print(f"Saved to {dest}")
        source = str(dest)

    elif os.path.isfile(source):
        # Local GGUF file — copy or symlink to models dir
        src_path = Path(source).resolve()
        dest = MODELS_DIR / src_path.name
        if not dest.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            os.symlink(src_path, dest)
            print(f"Linked {src_path.name} -> {dest}")
        if not name:
            name = src_path.stem.lower().replace(" ", "-")
        source = str(dest)

    else:
        print(f"Source not found: {source}", file=sys.stderr)
        sys.exit(1)

    # Detect model info with gguf metadata
    model_file = Path(source).name
    print(f"\nDetecting model properties...")
    info = _detect_gguf_info(source)

    # Pick a port (next available after existing)
    used_ports = {v.get("port", 0) for v in registry.values()}
    port = 8420
    while port in used_ports:
        port += 1

    # Register
    key = name or model_file.replace(".gguf", "").lower()
    registry[key] = {
        "name": info.get("name", key),
        "file": model_file,
        "binary": "turboquant-plus",
        "port": port,
        "context": info.get("context", 8192),
        "cache_k": "f16",
        "cache_v": "f16",
        "flash_attn": "on",
        "threads": 4,
        "notes": f"Added from {source}",
    }

    save_registry(registry)
    print(f"\nRegistered as '{key}':")
    print(f"  Name:    {registry[key]['name']}")
    print(f"  File:    {model_file}")
    print(f"  Port:    {port}")
    print(f"  Context: {registry[key]['context']}")
    print(f"\nStart with: local-model start {key}")


def cmd_info(args):
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    print(f"Model: {key}")
    print(f"  Name:     {cfg.get('name', '?')}")

    model_path = resolve_model_path(cfg)
    if model_path:
        print(f"  Path:     {model_path}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Size:     {size_mb:.0f} MB ({size_mb / 1024:.2f} GB)")
    elif cfg.get("source"):
        print(f"  Source:   {cfg['source']}")
    elif cfg.get("file"):
        print(f"  File:     {cfg['file']} (NOT FOUND)")

    print(f"  Binary:   {cfg.get('binary', '?')} -> {resolve_binary(cfg)}")
    print(f"  Port:     {cfg.get('port', '?')}")

    ctx = cfg.get("context", "?")
    if isinstance(ctx, int):
        print(f"  Context:  {ctx} ({ctx // 1024}K)")
    else:
        print(f"  Context:  {ctx}")

    print(f"  KV Cache: K={cfg.get('cache_k', '?')} V={cfg.get('cache_v', '?')}")
    print(f"  Flash:    {cfg.get('flash_attn', '?')}")
    print(f"  Threads:  {cfg.get('threads', '?')}")

    if cfg.get("notes"):
        print(f"  Notes:    {cfg['notes']}")

    pid = get_running_pid(key)
    if pid:
        healthy = check_health(cfg.get("port", 0))
        print(f"  Status:   Running (PID {pid}, {'healthy' if healthy else 'not ready'})")
    else:
        print(f"  Status:   Stopped")

    # Try to read GGUF metadata if we have the path
    if model_path and os.path.isfile(model_path):
        info = _detect_gguf_info(model_path)
        if info:
            print(f"\n  GGUF Metadata:")
            for k, v in sorted(info.items()):
                if k != "name":
                    print(f"    {k}: {v}")


# ── Helpers ──────────────────────────────────────────────────────────────────

_HAYSTACK_FILLER = (
    "The history of computing is filled with incremental advances that collectively "
    "transformed society. From Babbage's Analytical Engine to modern neural networks, "
    "each generation built upon the insights of its predecessors. Early vacuum tube "
    "computers filled entire rooms and consumed enormous amounts of power, yet their "
    "computational capacity was far less than a modern smartphone. "
)
_NEEDLE = "IMPORTANT FACT: The secret project codename is Operation Midnight Falcon."


def _build_haystack(target_tokens):
    target_chars = target_tokens * 4
    repeats = max(1, target_chars // len(_HAYSTACK_FILLER))
    parts = []
    mid = repeats // 2
    for i in range(repeats):
        if i == mid:
            parts.append(_NEEDLE)
        parts.append(_HAYSTACK_FILLER)
    return "\n\n".join(parts)[:target_chars]


def _chat(port, prompt, max_tokens=512, model_name="test"):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    elapsed = time.perf_counter() - t0
    msg = resp.get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    if not content and reasoning:
        content = reasoning[-300:]
    timings = resp.get("timings", {})
    usage = resp.get("usage", {})
    # llama.cpp: timings.prompt_per_second / predicted_per_second
    # mlx-vlm:  usage.prompt_tps / generation_tps
    # mlx-vlm 0.5.0 no longer includes those timing fields in the OpenAI
    # response, so fall back to wall-clock rates to keep table/test output useful.
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
    prompt_tps = timings.get("prompt_per_second") or usage.get("prompt_tps", 0)
    gen_tps = timings.get("predicted_per_second") or usage.get("generation_tps", 0)
    if not prompt_tps and elapsed and prompt_tokens:
        prompt_tps = prompt_tokens / elapsed
    if not gen_tps and elapsed and completion_tokens:
        gen_tps = completion_tokens / elapsed
    return {
        "content": content,
        "elapsed": round(elapsed, 2),
        "prompt_tokens": prompt_tokens,
        "tokens": completion_tokens,
        "gen_tps": round(gen_tps, 1),
        "prompt_tps": round(prompt_tps, 1),
    }


def _detect_gguf_info(path):
    """Read basic GGUF metadata using llama-cli or python."""
    info = {}
    # Try using a llama binary to dump metadata
    for bin_key, bin_path in BINARIES.items():
        cli = str(bin_path).replace("llama-server", "llama-gguf-hash")
        if not os.path.isfile(cli):
            continue
        # Fall back to checking with llama-server --verbose
        break

    # Simple binary header read for basic info
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return info
            import struct
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            info["format"] = f"GGUF v{version}"
            info["tensors"] = n_tensors
            info["metadata_entries"] = n_kv
    except Exception:
        pass

    return info


def cmd_help(args):
    print("local-model — manage local LLM inference servers\n")

    print("Commands:")
    print(f"  {'list':<28} Show available models and their status")
    print(f"  {'start <model> [--ctx N]':<28} Start a model server")
    print(f"  {'stop <model|all>':<28} Stop a running model server")
    print(f"  {'status':<28} Show running servers with health info")
    print(f"  {'test <model> [--prompts N]':<28} Run quality tests (reasoning, coding, factual, creative, needle)")
    print(f"  {'bench <model> [--ctx N]':<28} Run speed benchmark at multiple context sizes")
    print(f"  {'add <path|hf:repo> [name]':<28} Register a new GGUF model (local file or Hugging Face)")
    print(f"  {'info <model>':<28} Show model details (size, config, GGUF metadata)")
    print(f"  {'help':<28} Show this help")

    registry = load_registry()

    print(f"\nAvailable Models:")
    print(f"  {'Key':<14} {'Name':<26} {'Port':>5} {'Context':>8} {'tok/s in':>9} {'tok/s out':>9} {'Status'}")
    print(f"  {'-'*86}")

    for key, cfg in sorted(registry.items()):
        port = cfg.get("port", "?")
        ctx = cfg.get("context", "?")
        ctx_str = f"{ctx // 1024}K" if isinstance(ctx, int) and ctx >= 1024 else str(ctx)
        prompt_speed, gen_speed = _get_bench_speeds(key)
        in_str = f"{prompt_speed:.0f}" if prompt_speed else "—"
        out_str = f"{gen_speed:.1f}" if gen_speed else "—"

        pid = get_running_pid(key)
        if pid and check_health(port):
            status = "\033[32mrunning\033[0m"
        elif pid:
            status = "\033[33mstarting\033[0m"
        else:
            status = "\033[90mstopped\033[0m"

        model_path = resolve_model_path(cfg)
        if not model_path and not cfg.get("source", "").startswith("ollama:"):
            status = "\033[31mmissing\033[0m"

        print(f"  {key:<14} {cfg.get('name', '?'):<26} {port:>5} {ctx_str:>8} {in_str:>9} {out_str:>9} {status}")

        # Show notes indented underneath
        notes = cfg.get("notes", "")
        if notes:
            print(f"  {'':<14} \033[90m{notes}\033[0m")

    print(f"\nExamples:")
    print(f"  local-model start gemma4               Start Gemma 4 with default 128K context")
    print(f"  local-model start gemma4 --ctx 8192    Start with 8K context (faster startup, less RAM)")
    print(f"  local-model test gemma4                Run quality tests")
    print(f"  local-model bench gemma4               Run speed benchmark")
    print(f"  local-model stop all                   Stop all running servers")
    print(f"  local-model add hf:bartowski/Qwen3-8B-GGUF   Download and register from HF")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="local-model",
        description="Manage local LLM inference servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              local-model list                   Show all models and status
              local-model start gemma4           Start Gemma 4 E4B
              local-model start gemma4 --ctx 32768   Start with 32K context
              local-model stop all               Stop all running servers
              local-model test gemma4            Run quality tests
              local-model bench gemma4           Run speed benchmark
              local-model add hf:bartowski/Qwen3-8B-GGUF   Download and register
              local-model info nemotron          Show model details
        """),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="Show available models and their status")

    p = sub.add_parser("start", help="Start a model server")
    p.add_argument("model", help="Model name (e.g. gemma4, bonsai, nemotron)")
    p.add_argument("--ctx", type=int, help="Override context window size")

    p = sub.add_parser("stop", help="Stop a running model server")
    p.add_argument("model", help="Model name or 'all'")

    sub.add_parser("status", help="Show running servers with health info")

    p = sub.add_parser("test", help="Run quality tests against a running model")
    p.add_argument("model", help="Model name")
    p.add_argument("--prompts", type=int, help="Number of test prompts to run")

    p = sub.add_parser("bench", help="Run speed benchmark")
    p.add_argument("model", help="Model name")
    p.add_argument("--ctx", type=int, help="Max context to test")

    p = sub.add_parser("add", help="Register a new GGUF model")
    p.add_argument("source", help="Path to GGUF file or hf:<repo> for Hugging Face")
    p.add_argument("name", nargs="?", help="Short name for the model")

    p = sub.add_parser("info", help="Show model details")
    p.add_argument("model", help="Model name")

    sub.add_parser("help", help="Show commands and available models")

    args = parser.parse_args()

    if not args.command or args.command == "help":
        cmd_help(args)
        sys.exit(0)

    commands = {
        "list": cmd_list,
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "test": cmd_test,
        "bench": cmd_bench,
        "add": cmd_add,
        "info": cmd_info,
        "help": cmd_help,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
