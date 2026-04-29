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
  local-model config                      Show / edit configuration
"""
from __future__ import annotations
import argparse, json, os, platform, signal, shutil, struct, subprocess, sys
import textwrap, time, urllib.request, urllib.error
from pathlib import Path


# ── Paths & Config ─────────────────────────────────────────────────────────

def _home():
    """Resolve the local-model home directory."""
    return Path(os.environ.get("LOCAL_MODEL_HOME", Path.home() / ".local-model"))


ROOT = _home()
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
CONFIG_FILE = ROOT / "config.json"
REGISTRY_FILE = ROOT / "registry.json"


def _ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Backend Resolution ─────────────────────────────────────────────────────

def _load_config():
    """Load config.json (backends, defaults)."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_config(cfg):
    _ensure_dirs()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")


def _find_llama_server():
    """Find llama-server on PATH."""
    return shutil.which("llama-server")


def resolve_binary(model_cfg):
    """Resolve the server binary for a model.

    Resolution order:
    1. model_cfg["binary"] as absolute path
    2. model_cfg["binary"] as backend name in config.json backends
    3. "default" backend in config.json
    4. llama-server on PATH
    """
    bin_key = model_cfg.get("binary", "default")

    # Absolute path
    if os.path.isabs(bin_key) and os.path.isfile(bin_key):
        return bin_key

    # Named backend from config
    config = _load_config()
    backends = config.get("backends", {})

    if bin_key in backends:
        p = backends[bin_key]
        if os.path.isfile(p):
            return p

    # "default" backend
    if "default" in backends:
        p = backends["default"]
        if os.path.isfile(p):
            return p

    # Fall back to PATH
    found = _find_llama_server()
    if found:
        return found

    return None


# ── Model Registry ──────────────────────────────────────────────────────────

def load_registry():
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except Exception:
            pass
    return {}


def save_registry(registry):
    _ensure_dirs()
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2) + "\n")


def get_model(registry, name):
    if name in registry:
        return registry[name]
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


# ── Path Resolution ────────────────────────────────────────────────────────

def resolve_model_path(model_cfg):
    model_file = model_cfg.get("file", "")
    if model_file:
        # Check in MODELS_DIR
        p = MODELS_DIR / model_file
        if p.exists():
            return str(p)
        # Check as absolute path
        if os.path.isfile(model_file):
            return model_file

    # Check "dir" for directory-based models
    model_dir = model_cfg.get("dir", "")
    if model_dir:
        p = MODELS_DIR / model_dir
        if p.is_dir():
            return str(p)
        if os.path.isdir(model_dir):
            return model_dir

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
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pf.unlink(missing_ok=True)
        return None


def check_health(port, timeout=3):
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout)
        status = json.loads(r.read()).get("status", "")
        return status in ("ok", "healthy")
    except Exception:
        return False


# ── Server Command Builder ─────────────────────────────────────────────────

def _build_server_cmd(cfg, binary, model_path, port, ctx):
    cmd = [
        binary, "-m", model_path,
        "-ngl", str(cfg.get("gpu_layers", 99)),
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
        elif os.path.isfile(mmproj):
            cmd += ["-mm", mmproj]
        else:
            print(f"  warning: mmproj not found: {mmproj}", file=sys.stderr)

    # Pass through extra server args from config
    extra = cfg.get("server_args", [])
    if extra:
        cmd += extra

    return cmd


def _describe_config(cfg):
    parts = [f"KV: {cfg.get('cache_k', 'f16')}/{cfg.get('cache_v', 'f16')}"]
    if cfg.get("flash_attn") == "off":
        parts.append("no flash_attn")
    if cfg.get("gpu_layers", 99) == 0:
        parts.append("CPU only")
    return "  ".join(parts)


# ── Bench Speeds (for dashboard) ───────────────────────────────────────────

def _get_bench_speeds(key):
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


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_list(args):
    registry = load_registry()
    if not registry:
        print("No models registered. Add one with:")
        print("  local-model add <path-to-gguf>")
        print("  local-model add hf:<huggingface-repo>")
        return

    print(f"{'Model':<18} {'Name':<28} {'Port':>5} {'Context':>8} {'tok/s in':>9} {'tok/s out':>9} {'Status':<10}")
    print("-" * 95)
    for key, cfg in sorted(registry.items()):
        port = cfg.get("port", "?")
        ctx = cfg.get("context", "?")
        if isinstance(ctx, int):
            ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        else:
            ctx_str = str(ctx)

        prompt_speed, gen_speed = _get_bench_speeds(key)
        in_str = f"{prompt_speed:.0f}" if prompt_speed else "-"
        out_str = f"{gen_speed:.1f}" if gen_speed else "-"

        pid = get_running_pid(key)
        if pid and check_health(port):
            status = f"\033[32mrunning\033[0m (:{port})"
        elif pid:
            status = f"\033[33mstarting\033[0m"
        else:
            status = "\033[90mstopped\033[0m"

        model_path = resolve_model_path(cfg)
        if not model_path:
            status = "\033[31mmissing\033[0m"

        print(f"{key:<18} {cfg.get('name', '?'):<28} {port:>5} {ctx_str:>8} {in_str:>9} {out_str:>9} {status}")


def cmd_start(args):
    _ensure_dirs()
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    pid = get_running_pid(key)
    if pid and check_health(cfg["port"]):
        print(f"{cfg['name']} is already running on port {cfg['port']} (PID {pid})")
        return

    port = cfg.get("port", 8080)

    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except ProcessLookupError:
            pass

    binary = resolve_binary(cfg)
    if not binary or not os.path.isfile(binary):
        print(f"Server binary not found.", file=sys.stderr)
        print(f"Configure a backend with: local-model config --set-backend default /path/to/llama-server", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model_path(cfg)
    if not model_path:
        print(f"Model file not found for '{key}'.", file=sys.stderr)
        sys.exit(1)

    ctx = args.ctx or cfg.get("context", 8192)
    cmd = _build_server_cmd(cfg, binary, model_path, port, ctx)

    log_f = log_file_for(key)
    print(f"Starting {cfg['name']}...")
    print(f"  port: {port}  ctx: {ctx}  {_describe_config(cfg)}")

    with open(log_f, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

    pid_file_for(key).write_text(str(proc.pid))
    print(f"  pid: {proc.pid}  log: {log_f}")

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


# ── Test & Bench ────────────────────────────────────────────────────────────

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
    prompt_tps = timings.get("prompt_per_second") or usage.get("prompt_tps", 0)
    gen_tps = timings.get("predicted_per_second") or usage.get("generation_tps", 0)
    return {
        "content": content,
        "elapsed": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens", 0),
        "tokens": usage.get("completion_tokens") or usage.get("output_tokens", 0),
        "gen_tps": round(gen_tps, 1),
        "prompt_tps": round(prompt_tps, 1),
    }


def _ensure_clean_for_bench(registry, target_key):
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
    cfg = registry[key]
    port = cfg.get("port", 8080)

    pid = get_running_pid(key)
    if pid and check_health(port):
        return port, False

    binary = resolve_binary(cfg)
    if not binary or not os.path.isfile(binary):
        print(f"Server binary not found. Configure with: local-model config --set-backend default /path/to/llama-server", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model_path(cfg)
    if not model_path:
        print(f"Model file not found for {key}", file=sys.stderr)
        sys.exit(1)

    ctx = ctx_override or cfg.get("context", 8192)

    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except ProcessLookupError:
            pass

    cmd = _build_server_cmd(cfg, binary, model_path, port, ctx)

    _ensure_dirs()
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


def cmd_test(args):
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    _ensure_clean_for_bench(registry, key)
    port, started_by_us = _start_for_bench(registry, key)

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
            r = _chat(port, prompt, max_tok)
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


def cmd_bench(args):
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    _ensure_clean_for_bench(registry, key)

    ctx_override = args.ctx or cfg.get("context", 8192)
    port, started_by_us = _start_for_bench(registry, key, ctx_override)

    actual_ctx = None
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=3)
        slots = json.loads(r.read())
        if slots:
            actual_ctx = slots[0].get("n_ctx")
    except Exception:
        pass
    ctx_limit = actual_ctx or ctx_override

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
            r = _chat(port, text, max_tokens=128)
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

    if started_by_us:
        pid = get_running_pid(key)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            pid_file_for(key).unlink(missing_ok=True)
            print(f"\n{cfg['name']} stopped (was started for benchmark).")


# ── Add Model ───────────────────────────────────────────────────────────────

def _detect_gguf_info(path):
    info = {}
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return info
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            info["format"] = f"GGUF v{version}"
            info["tensors"] = n_tensors
            info["metadata_entries"] = n_kv
    except Exception:
        pass
    return info


def cmd_add(args):
    registry = load_registry()
    source = args.source
    name = args.name

    if source.startswith("hf:") or ("/" in source and not os.path.exists(source)):
        hf_repo = source.replace("hf:", "")
        if not name:
            name = hf_repo.split("/")[-1].lower().replace(" ", "-")

        print(f"Downloading from Hugging Face: {hf_repo}")
        print(f"Looking for GGUF files...")

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
        _ensure_dirs()
        dest = MODELS_DIR / gguf_name
        dl_url = f"https://huggingface.co/{hf_repo}/resolve/main/{gguf_name}"

        print(f"\nDownloading {gguf_name}...")
        subprocess.run(["curl", "-L", "-o", str(dest), "--progress-bar", dl_url], check=True)
        print(f"Saved to {dest}")
        source = str(dest)

    elif os.path.isfile(source):
        src_path = Path(source).resolve()
        _ensure_dirs()
        dest = MODELS_DIR / src_path.name
        if not dest.exists():
            os.symlink(src_path, dest)
            print(f"Linked {src_path.name} -> {dest}")
        if not name:
            name = src_path.stem.lower().replace(" ", "-")
        source = str(dest)

    else:
        print(f"Source not found: {source}", file=sys.stderr)
        sys.exit(1)

    model_file = Path(source).name
    print(f"\nDetecting model properties...")
    info = _detect_gguf_info(source)

    used_ports = {v.get("port", 0) for v in registry.values()}
    port = 8080
    while port in used_ports:
        port += 1

    key = name or model_file.replace(".gguf", "").lower()
    registry[key] = {
        "name": info.get("name", key),
        "file": model_file,
        "binary": "default",
        "port": port,
        "context": info.get("context", 8192),
        "cache_k": "f16",
        "cache_v": "f16",
        "flash_attn": "on",
        "threads": 4,
        "notes": f"Added from {Path(source).name}",
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
    if model_path and os.path.isfile(model_path):
        print(f"  Path:     {model_path}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Size:     {size_mb:.0f} MB ({size_mb / 1024:.2f} GB)")
    elif cfg.get("file"):
        print(f"  File:     {cfg['file']} (NOT FOUND)")

    binary = resolve_binary(cfg)
    print(f"  Binary:   {binary or 'not configured'}")
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

    if model_path and os.path.isfile(model_path):
        info = _detect_gguf_info(model_path)
        if info:
            print(f"\n  GGUF Metadata:")
            for k, v in sorted(info.items()):
                if k != "name":
                    print(f"    {k}: {v}")


# ── Config ──────────────────────────────────────────────────────────────────

def cmd_config(args):
    config = _load_config()

    if args.set_backend:
        name, path = args.set_backend
        path = str(Path(path).resolve())
        if not os.path.isfile(path):
            print(f"Warning: {path} does not exist yet", file=sys.stderr)
        backends = config.setdefault("backends", {})
        backends[name] = path
        _save_config(config)
        print(f"Backend '{name}' -> {path}")
        return

    if args.set_threads:
        config["default_threads"] = args.set_threads
        _save_config(config)
        print(f"Default threads: {args.set_threads}")
        return

    # Show current config
    print(f"Home:     {ROOT}")
    print(f"Models:   {MODELS_DIR}")
    print(f"Logs:     {LOGS_DIR}")
    print(f"Config:   {CONFIG_FILE}")
    print(f"Registry: {REGISTRY_FILE}")

    backends = config.get("backends", {})
    if backends:
        print(f"\nBackends:")
        for name, path in sorted(backends.items()):
            exists = "OK" if os.path.isfile(path) else "NOT FOUND"
            print(f"  {name:<16} {path}  [{exists}]")
    else:
        found = _find_llama_server()
        print(f"\nBackends: none configured")
        if found:
            print(f"  (llama-server found on PATH: {found})")
        else:
            print(f"  Configure with: local-model config --set-backend default /path/to/llama-server")

    print(f"\nPlatform: {platform.system()} {platform.machine()}")


# ── Help ────────────────────────────────────────────────────────────────────

def cmd_help(args):
    print("local-model — manage local LLM inference servers\n")

    print("Commands:")
    print(f"  {'list':<30} Show available models and their status")
    print(f"  {'start <model> [--ctx N]':<30} Start a model server")
    print(f"  {'stop <model|all>':<30} Stop a running model server")
    print(f"  {'status':<30} Show running servers with health info")
    print(f"  {'test <model> [--prompts N]':<30} Run quality tests (reasoning, coding, factual)")
    print(f"  {'bench <model> [--ctx N]':<30} Run speed benchmark at multiple context sizes")
    print(f"  {'add <path|hf:repo> [name]':<30} Register a new GGUF model")
    print(f"  {'info <model>':<30} Show model details (size, config, GGUF metadata)")
    print(f"  {'config':<30} Show configuration and backend paths")
    print(f"  {'config --set-backend N path':<30} Configure a named backend binary")
    print(f"  {'help':<30} Show this help")

    registry = load_registry()
    if registry:
        print(f"\nRegistered Models:")
        print(f"  {'Key':<18} {'Name':<28} {'Port':>5}")
        print(f"  {'-'*55}")
        for key, cfg in sorted(registry.items()):
            print(f"  {key:<18} {cfg.get('name', '?'):<28} {cfg.get('port', '?'):>5}")

    print(f"\nQuick Start:")
    print(f"  1. Install a llama-server binary (llama.cpp, PrismML, etc.)")
    print(f"  2. local-model config --set-backend default /path/to/llama-server")
    print(f"  3. local-model add hf:prism-ml/Ternary-Bonsai-8B-gguf")
    print(f"  4. local-model start ternary-bonsai-8b-gguf")
    print(f"\nEnvironment:")
    print(f"  LOCAL_MODEL_HOME  Override home directory (default: ~/.local-model)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="local-model",
        description="Manage local LLM inference servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              local-model list                              Show all models and status
              local-model add hf:prism-ml/Ternary-Bonsai-8B-gguf   Download from HF
              local-model start bonsai                      Start a model server
              local-model stop all                          Stop all running servers
              local-model test bonsai                       Run quality tests
              local-model bench bonsai                      Run speed benchmark
              local-model config --set-backend default /usr/local/bin/llama-server
        """),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="Show available models and their status")

    p = sub.add_parser("start", help="Start a model server")
    p.add_argument("model", help="Model name")
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

    p = sub.add_parser("config", help="Show / edit configuration")
    p.add_argument("--set-backend", nargs=2, metavar=("NAME", "PATH"),
                    help="Set a named backend binary path")
    p.add_argument("--set-threads", type=int, metavar="N",
                    help="Set default thread count")

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
        "config": cmd_config,
        "help": cmd_help,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
