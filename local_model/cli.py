#!/usr/bin/env python3
"""local-model — manage local LLM inference servers.

Usage:
  local-model list                        Show available models and their status
  local-model start <model> [--ctx N]     Start a model server
  local-model serve <model> [--lan]       Expose a model over Tailscale/LAN
  local-model stop <model|all>            Stop a running model server
  local-model status                      Show running servers with health + memory
  local-model monitor                     Show RAM/VRAM attribution bars
  local-model scan                        Scan the LAN for served models
  local-model test <model> [--prompts N]  Run quality tests against a running model
  local-model bench <model> [--ctx N]     Run speed benchmark
  local-model add <path|hf-repo> [name]   Register a new GGUF model
  local-model info <model>                Show model details (arch, params, quant, ctx)
  local-model config                      Show / edit configuration
"""
from __future__ import annotations
import argparse, concurrent.futures, ipaddress, json, os, platform, re, signal, shutil
import socket, struct, subprocess, sys
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


def _pid_alive(pid):
    """Cross-platform liveness check that does NOT kill the process.

    On Windows, os.kill(pid, 0) calls TerminateProcess and would KILL the
    target, so we use OpenProcess + GetExitCodeProcess instead. On POSIX,
    signal 0 is a real liveness probe.
    """
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return False
        finally:
            k32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but owned by another user
        return True


def _terminate_pid(pid):
    """Cross-platform termination. llama-server is stateless (in-memory KV
    cache, nothing to flush) so a force kill is safe. Windows: taskkill /T to
    also reap children. POSIX: SIGTERM.
    """
    if not pid:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def get_running_pid(key):
    pf = pid_file_for(key)
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text().strip())
    except ValueError:
        pf.unlink(missing_ok=True)
        return None
    if _pid_alive(pid):
        return pid
    pf.unlink(missing_ok=True)
    return None


def _is_remote(cfg):
    return bool(cfg.get("remote"))


def _model_endpoint(cfg):
    """OpenAI base URL (.../v1) for a model: remote 'url' if set, else local port."""
    if _is_remote(cfg) and cfg.get("url"):
        return cfg["url"].rstrip("/")
    return f"http://127.0.0.1:{cfg.get('port', 8080)}/v1"


def _check_endpoint(cfg, timeout=5):
    """Reachability via GET <endpoint>/models. Works for local and remote."""
    try:
        r = urllib.request.urlopen(_model_endpoint(cfg) + "/models", timeout=timeout)
        return r.getcode() == 200
    except Exception:
        return False


def _normalize_remote_url(u):
    """Accept host:port or a full URL; normalize to a .../v1 base URL."""
    u = u.strip().rstrip("/")
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def _parse_port_list(text):
    ports = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            try:
                first, last = int(start), int(end)
            except ValueError:
                raise ValueError(f"invalid port range '{part}'")
            if first > last:
                first, last = last, first
            values = range(first, last + 1)
        else:
            try:
                values = [int(part)]
            except ValueError:
                raise ValueError(f"invalid port '{part}'")
        for port in values:
            if port < 1 or port > 65535:
                raise ValueError(f"port out of range: {port}")
            if port not in ports:
                ports.append(port)
    if not ports:
        raise ValueError("at least one port is required")
    return ports


def _local_lan_cidr():
    host = _local_ip_address()
    if host and not host.startswith("127."):
        return f"{host}/24"
    return None


def _local_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None
    return None


def _default_scan_targets():
    targets = ["127.0.0.1"]
    lan = _local_lan_cidr()
    if lan:
        targets.append(lan)
    return targets


def _expand_scan_targets(targets):
    seen = set()
    hosts = []
    for target in targets:
        target = (target or "").strip()
        if not target:
            continue
        if "/" in target:
            try:
                net = ipaddress.ip_network(target, strict=False)
            except ValueError as exc:
                raise ValueError(f"invalid network target '{target}': {exc}")
            iterable = net.hosts() if net.num_addresses > 2 else net
            for addr in iterable:
                host = str(addr)
                if host not in seen:
                    seen.add(host)
                    hosts.append(host)
        elif target not in seen:
            seen.add(target)
            hosts.append(target)
    return hosts


def _models_from_payload(payload):
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if data is None:
        data = payload.get("models")
    if not isinstance(data, list):
        return []
    models = []
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("model")
        else:
            model_id = item
        if model_id:
            models.append(str(model_id))
    return models


def _probe_openai_models(host, port, timeout):
    base = f"http://{host}:{port}/v1"
    payload = _http_json(base + "/models", timeout=timeout)
    models = _models_from_payload(payload)
    if not models:
        return None
    return {
        "host": host,
        "port": port,
        "url": base,
        "models": models,
    }


def _safe_key(value, fallback="remote"):
    key = re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-._")
    return key or fallback


def _unique_key(registry, base_key):
    key = base_key
    i = 2
    while key in registry:
        key = f"{base_key}-{i}"
        i += 1
    return key


def _lan_model_url(port):
    lan_ip = _local_ip_address()
    if not lan_ip or lan_ip.startswith("127."):
        return None
    return f"http://{lan_ip}:{port}/v1"


def _check_lan_endpoint(port, timeout=1.0):
    url = _lan_model_url(port)
    if not url:
        return False
    return bool(_models_from_payload(_http_json(url + "/models", timeout=timeout)))


def check_health(port, timeout=3):
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout)
        status = json.loads(r.read()).get("status", "")
        return status in ("ok", "healthy")
    except Exception:
        return False


# ── External Providers ─────────────────────────────────────────────────────

OLLAMA_URL = "http://127.0.0.1:11434"
WHISPER_ROOT = Path(os.environ.get("WHISPER_CPP_ROOT", Path.home() / "whisper.cpp"))
WHISPER_MODELS = WHISPER_ROOT / "models"
WHISPER_DEFAULT_PORT = 8178
KOKORO_ROOT = Path(os.environ.get("KOKORO_FASTAPI_ROOT", r"X:\Local-Model\kokoro-fastapi"))
KOKORO_MODEL = KOKORO_ROOT / "api" / "src" / "models" / "v1_0" / "kokoro-v1_0.pth"
KOKORO_DEFAULT_PORT = 8880
BONSAI_IMAGE_ROOT = Path(os.environ.get("BONSAI_IMAGE_ROOT", r"X:\Local-Model\bonsai-image-gemlite"))
BONSAI_IMAGE_DEFAULT_PORT = 8000
VOXCPM_ROOT = Path(os.environ.get("VOXCPM_ROOT", r"X:\Local-Model\VoxCPM"))
VOXCPM_MODEL_DIR = Path(os.environ.get("VOXCPM_MODEL_PATH", r"X:\Local-Model\models\openbmb__VoxCPM2"))
VOXCPM_DEFAULT_PORT = 8808


def _http_json(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _http_text(url, timeout=1.5, max_bytes=65536):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(max_bytes)
            charset = r.headers.get_content_charset() or "utf-8"
            return r.getcode(), body.decode(charset, errors="replace")
    except Exception:
        return None, ""


def _fmt_bytes(num):
    if num in (None, ""):
        return "-"
    try:
        num = float(num)
    except (TypeError, ValueError):
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(num) < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(num)} {unit}"
            return f"{num:.1f} {unit}"
        num /= 1024
    return "-"


def _dir_size(path):
    if not path or not Path(path).is_dir():
        return None
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except OSError:
        return None
    return total


def _process_snapshots():
    """Best-effort process table with pid/name/command/working-set bytes."""
    if os.name == "nt":
        ps = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,Name,CommandLine,WorkingSetSize | "
            "ConvertTo-Json -Compress"
        )
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=10)
            data = json.loads(r.stdout or "[]")
            if isinstance(data, dict):
                data = [data]
            out = {}
            for p in data:
                pid = p.get("ProcessId")
                if pid is None:
                    continue
                out[int(pid)] = {
                    "pid": int(pid),
                    "name": p.get("Name") or "",
                    "cmd": p.get("CommandLine") or "",
                    "rss": int(p.get("WorkingSetSize") or 0),
                }
            return out
        except Exception:
            return {}

    try:
        r = subprocess.run(["ps", "-eo", "pid=,rss=,comm=,args="],
                           capture_output=True, text=True, timeout=10)
        out = {}
        for line in r.stdout.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 3:
                continue
            pid, rss_kb, name = parts[:3]
            cmd = parts[3] if len(parts) > 3 else name
            out[int(pid)] = {
                "pid": int(pid),
                "name": name,
                "cmd": cmd,
                "rss": int(rss_kb) * 1024,
            }
        return out
    except Exception:
        return {}


def _system_memory():
    """Return total/used RAM bytes."""
    if os.name == "nt":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys), int(stat.ullTotalPhys - stat.ullAvailPhys)
        except Exception:
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        avail = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = pages * page_size
        used = (pages - avail) * page_size
        return int(total), int(used)
    except Exception:
        return None, None


def _gpu_memory(target_pids=None):
    """Return total/used/free VRAM bytes and per-process VRAM bytes."""
    target_pids = {int(pid) for pid in (target_pids or []) if pid}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        total_mb, used_mb, free_mb = [
            int(x.strip()) for x in r.stdout.strip().splitlines()[0].split(",")
        ]
    except Exception:
        return None, None, None, {}

    per_pid = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not parts[0].isdigit() or not parts[2].isdigit():
                continue
            pid = int(parts[0])
            if target_pids and pid not in target_pids:
                continue
            per_pid[pid] = {
                "name": Path(parts[1]).name,
                "vram": int(parts[2]) * 1024 * 1024,
            }
    except Exception:
        pass

    missing_pids = target_pids - set(per_pid)
    if os.name == "nt" and missing_pids:
        per_pid.update(_windows_gpu_process_memory(missing_pids))
    return total_mb * 1024 * 1024, used_mb * 1024 * 1024, free_mb * 1024 * 1024, per_pid


def _windows_gpu_adapter_used():
    ps = (
        "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples | "
        "Select-Object InstanceName,CookedValue | ConvertTo-Json -Compress"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=5)
        data = json.loads(r.stdout or "[]")
        if isinstance(data, dict):
            data = [data]
    except Exception:
        return None

    values = []
    for sample in data:
        try:
            values.append(int(float(sample.get("CookedValue") or 0)))
        except (TypeError, ValueError):
            pass
    return max(values) if values else None


def _windows_gpu_process_memory(target_pids=None):
    target_pids = {int(pid) for pid in (target_pids or []) if pid}
    ps = (
        "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples | "
        "Where-Object { $_.CookedValue -gt 0 } | "
        "Select-Object InstanceName,CookedValue | ConvertTo-Json -Compress"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=5)
        data = json.loads(r.stdout or "[]")
        if isinstance(data, dict):
            data = [data]
    except Exception:
        return {}

    out = {}
    for sample in data:
        m = re.search(r"pid_(\d+)_", sample.get("InstanceName", ""))
        if not m:
            continue
        try:
            value = int(float(sample.get("CookedValue") or 0))
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        pid = int(m.group(1))
        if target_pids and pid not in target_pids:
            continue
        item = out.setdefault(pid, {"name": "", "vram": 0})
        item["vram"] += value
    return out


def _port_owner_pid(port):
    """Best-effort listener owner lookup for local services."""
    try:
        port = int(port)
    except (TypeError, ValueError):
        return None

    if os.name == "nt":
        ps = (
            f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen "
            "-ErrorAction SilentlyContinue | Select-Object -First 1; "
            "if ($c) { $c.OwningProcess }"
        )
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=3)
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    return int(line)
        except Exception:
            return None
    return None


def _discover_ollama():
    tags = _http_json(OLLAMA_URL + "/api/tags") or {}
    ps = _http_json(OLLAMA_URL + "/api/ps") or {}
    installed = {m.get("name") or m.get("model"): m
                 for m in tags.get("models", []) if m.get("name") or m.get("model")}
    running = {m.get("name") or m.get("model"): m
               for m in ps.get("models", []) if m.get("name") or m.get("model")}
    rows = []
    for name in sorted(set(installed) | set(running)):
        meta = running.get(name) or installed.get(name) or {}
        details = meta.get("details") or {}
        ctx = details.get("context_length") or "?"
        q = details.get("quantization_level")
        family = details.get("family")
        display = name
        if family or q:
            display += f" ({'/'.join(x for x in [family, q] if x)})"
        rows.append({
            "key": f"ollama:{name}",
            "source": "ollama",
            "kind": "llm",
            "name": display,
            "port": 11434,
            "context": ctx,
            "size": meta.get("size"),
            "ram": None,
            "vram": meta.get("size_vram") if name in running else None,
            "status": "running" if name in running else "available",
        })
    return rows


def _whisper_processes(processes=None):
    processes = processes or _process_snapshots()
    out = []
    for p in processes.values():
        name = (p.get("name") or "").lower()
        cmd = (p.get("cmd") or "").lower()
        if "whisper-server" in name or "whisper-server" in cmd:
            out.append(p)
    return out


def _parse_arg_value(cmd, flag):
    if not cmd:
        return None
    m = re.search(rf"{re.escape(flag)}\s+\"([^\"]+)\"", cmd)
    if m:
        return m.group(1)
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", cmd)
    return m.group(1) if m else None


def _discover_whisper(processes=None):
    processes = processes or _process_snapshots()
    running = _whisper_processes(processes)
    running_by_model = {}
    for p in running:
        model = _parse_arg_value(p.get("cmd", ""), "-m")
        if model:
            running_by_model[Path(model).name] = p

    model_files = []
    if WHISPER_MODELS.is_dir():
        model_files = [
            p for p in WHISPER_MODELS.glob("*.bin")
            if not p.name.startswith("for-tests-")
        ]

    if not model_files and running:
        model_files = [Path(_parse_arg_value(running[0].get("cmd", ""), "-m") or "whisper.cpp")]

    rows = []
    for model in sorted(model_files, key=lambda p: p.name.lower()):
        proc = running_by_model.get(model.name) or (running[0] if running else None)
        port = _parse_arg_value(proc.get("cmd", ""), "--port") if proc else WHISPER_DEFAULT_PORT
        rows.append({
            "key": f"whisper:{model.stem}",
            "source": "whisper.cpp",
            "kind": "stt",
            "name": model.name,
            "port": int(port) if str(port).isdigit() else port,
            "context": "-",
            "size": model.stat().st_size if model.exists() else None,
            "ram": proc.get("rss") if proc else None,
            "vram": None,
            "pid": proc.get("pid") if proc else None,
            "status": "running" if proc else "available",
        })
    return rows


def _looks_like_kokoro_process(proc):
    cmd = (proc.get("cmd") or "").lower()
    name = (proc.get("name") or "").lower()
    return (
        "kokoro" in name
        or "kokoro-fastapi" in cmd
        or "run-kokoro" in cmd
        or ("uvicorn" in cmd and "api.src.main:app" in cmd)
    )


def _kokoro_processes(processes=None):
    processes = processes or _process_snapshots()
    return [p for p in processes.values() if _looks_like_kokoro_process(p)]


def _pick_likely_model_process(candidates):
    return max(candidates, key=lambda p: p.get("rss") or 0) if candidates else None


def _kokoro_reachable(port, timeout=1.5):
    health = _http_json(f"http://127.0.0.1:{port}/health", timeout=timeout)
    if isinstance(health, dict):
        status = str(health.get("status", "")).lower()
        if status in ("ok", "healthy"):
            return True

    models = _http_json(f"http://127.0.0.1:{port}/v1/models", timeout=timeout)
    return isinstance(models, dict) and bool(models.get("data") or models.get("object"))


def _discover_kokoro(processes=None, fast=False):
    processes = processes or _process_snapshots()
    port = KOKORO_DEFAULT_PORT
    candidates = _kokoro_processes(processes)
    proc = _pick_likely_model_process(candidates)
    pid = proc.get("pid") if proc else None

    if proc:
        parsed_port = _parse_arg_value(proc.get("cmd", ""), "--port")
        if parsed_port and str(parsed_port).isdigit():
            port = int(parsed_port)
    elif not fast:
        pid = _port_owner_pid(port)
        proc = processes.get(pid) if pid else None

    if proc and not _looks_like_kokoro_process(proc) and not _kokoro_reachable(port):
        proc = None
        pid = None

    should_probe = bool(proc or pid or (not fast and os.name != "nt"))
    reachable = _kokoro_reachable(port, timeout=0.25 if fast else 1.5) if should_probe else False
    if not (KOKORO_ROOT.exists() or proc or reachable):
        return []

    if reachable:
        status = "running"
    elif proc:
        status = "starting"
    elif KOKORO_MODEL.exists():
        status = "available"
    else:
        status = "missing"

    return [{
        "key": "kokoro:kokoro-v1_0",
        "source": "kokoro",
        "kind": "tts",
        "name": KOKORO_MODEL.name,
        "port": port,
        "context": "-",
        "size": KOKORO_MODEL.stat().st_size if KOKORO_MODEL.exists() else None,
        "ram": proc.get("rss") if proc else None,
        "vram": None,
        "pid": pid,
        "status": status,
    }]


def _looks_like_bonsai_image_process(proc):
    cmd = (proc.get("cmd") or "").lower()
    return (
        "bonsai-image-gemlite" in cmd
        or "scripts.local_backend:app" in cmd
        or "local_backend.py" in cmd
    )


def _bonsai_image_processes(processes=None):
    processes = processes or _process_snapshots()
    return [p for p in processes.values() if _looks_like_bonsai_image_process(p)]


def _bonsai_image_backend(port, timeout=1.5):
    info = _http_json(f"http://127.0.0.1:{port}/backends", timeout=timeout)
    return info if isinstance(info, dict) else None


def _bonsai_image_model_dir(backend=None):
    if backend:
        family = str(backend.get("default_family") or "").replace("bonsai-", "")
        kind = str(backend.get("kind") or "gemlite")
        if family:
            path = BONSAI_IMAGE_ROOT / "models" / f"bonsai-image-4B-{family}-{kind}"
            if path.is_dir():
                return path

    models = BONSAI_IMAGE_ROOT / "models"
    if not models.is_dir():
        return None
    matches = sorted(models.glob("bonsai-image-4B-*-gemlite"))
    return matches[0] if matches else None


def _discover_bonsai_image(processes=None, fast=False):
    processes = processes or _process_snapshots()
    port = BONSAI_IMAGE_DEFAULT_PORT
    candidates = _bonsai_image_processes(processes)
    backend_candidates = [
        p for p in candidates
        if "scripts.local_backend:app" in (p.get("cmd") or "").lower()
        or "local_backend.py" in (p.get("cmd") or "").lower()
    ]
    proc = _pick_likely_model_process(backend_candidates or candidates)
    pid = proc.get("pid") if proc else None

    if proc:
        parsed_port = _parse_arg_value(proc.get("cmd", ""), "--port")
        if parsed_port and str(parsed_port).isdigit():
            port = int(parsed_port)
    elif not fast:
        pid = _port_owner_pid(port)
        proc = processes.get(pid) if pid else None

    if proc and not _looks_like_bonsai_image_process(proc) and not _bonsai_image_backend(port):
        proc = None
        pid = None

    should_probe = bool(proc or pid or (not fast and os.name != "nt"))
    backend = _bonsai_image_backend(port, timeout=0.25 if fast else 1.5) if should_probe else None
    if not (BONSAI_IMAGE_ROOT.exists() or proc or backend):
        return []

    model_dir = _bonsai_image_model_dir(backend)
    if backend and backend.get("healthy"):
        status = "running"
    elif proc:
        status = "starting"
    elif model_dir:
        status = "available"
    else:
        status = "missing"

    family = str((backend or {}).get("default_family") or "bonsai-image")
    kind = str((backend or {}).get("kind") or "gemlite")
    model_id = family if family.endswith(f"-{kind}") else f"{family}-{kind}"

    return [{
        "key": f"bonsai-image:{model_id}",
        "source": "bonsai-image",
        "kind": "image",
        "name": model_dir.name if model_dir else "Bonsai Image 4B",
        "port": port,
        "context": "-",
        "size": _dir_size(model_dir),
        "ram": proc.get("rss") if proc else None,
        "vram": None,
        "pid": pid,
        "status": status,
    }]


def _looks_like_voxcpm_process(proc):
    cmd = (proc.get("cmd") or "").lower()
    return (
        "voxcpm" in cmd
        or str(VOXCPM_ROOT).lower() in cmd
        or ("app.py" in cmd and f"--port {VOXCPM_DEFAULT_PORT}" in cmd)
    )


def _voxcpm_processes(processes=None):
    processes = processes or _process_snapshots()
    return [p for p in processes.values() if _looks_like_voxcpm_process(p)]


def _voxcpm_reachable(port, timeout=1.5):
    config = _http_json(f"http://127.0.0.1:{port}/config", timeout=timeout)
    if isinstance(config, dict):
        text = json.dumps(config).lower()
        if "voxcpm" in text:
            return True

    code, text = _http_text(f"http://127.0.0.1:{port}/", timeout=timeout)
    return code == 200 and "voxcpm" in text.lower()


def _discover_voxcpm(processes=None, fast=False):
    processes = processes or _process_snapshots()
    port = VOXCPM_DEFAULT_PORT
    candidates = _voxcpm_processes(processes)
    proc = _pick_likely_model_process(candidates)
    pid = proc.get("pid") if proc else None

    if proc:
        parsed_port = _parse_arg_value(proc.get("cmd", ""), "--port")
        if parsed_port and str(parsed_port).isdigit():
            port = int(parsed_port)
    elif not fast:
        pid = _port_owner_pid(port)
        proc = processes.get(pid) if pid else None

    if proc and not _looks_like_voxcpm_process(proc) and not _voxcpm_reachable(port):
        proc = None
        pid = None

    should_probe = bool(proc or pid or (not fast and os.name != "nt"))
    reachable = _voxcpm_reachable(port, timeout=0.25 if fast else 1.5) if should_probe else False
    if not (VOXCPM_ROOT.exists() or VOXCPM_MODEL_DIR.exists() or proc or reachable):
        return []

    if reachable:
        status = "running"
    elif proc:
        status = "starting"
    elif VOXCPM_MODEL_DIR.exists():
        status = "available"
    else:
        status = "missing"

    return [{
        "key": "voxcpm:voxcpm2",
        "source": "voxcpm",
        "kind": "tts",
        "name": VOXCPM_MODEL_DIR.name if VOXCPM_MODEL_DIR.exists() else "VoxCPM2",
        "port": port,
        "context": "-",
        "size": _dir_size(VOXCPM_MODEL_DIR),
        "ram": proc.get("rss") if proc else None,
        "vram": None,
        "pid": pid,
        "status": status,
    }]


def _external_rows(processes=None, fast=False):
    return (_discover_ollama() + _discover_whisper(processes)
            + _discover_kokoro(processes, fast=fast)
            + _discover_bonsai_image(processes, fast=fast)
            + _discover_voxcpm(processes, fast=fast))



# ── Server Command Builder ─────────────────────────────────────────────────

def _query_free_vram_mb():
    """Free VRAM (MiB) of the first GPU via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def _compute_ncmoe(cfg, ctx, quiet=False):
    """VRAM-aware -ncmoe (ported from run-qwen.ps1).

    Active only when the model config has an "auto_ncmoe" block. Computes how
    many MoE expert layers to keep on CPU based on free VRAM at start time, so
    it adapts to whatever else is on the GPU. Returns int, or None to skip.

    auto_ncmoe may override any calibration constant; defaults are calibrated
    for Qwen3.6-35B-A3B IQ3_XXS + turbo3 KV on a 12 GB card.
    """
    auto = cfg.get("auto_ncmoe")
    if not auto:
        return None
    free_mb = _query_free_vram_mb()
    if free_mb is None:
        if not quiet:
            print("  warning: nvidia-smi unavailable; skipping auto-ncmoe",
                  file=sys.stderr)
        return None
    total   = int(auto.get("total_layers", 40))
    per     = float(auto.get("per_layer_expert_mb", 310))
    base    = float(auto.get("base_gpu_mb", 900))
    compute = float(auto.get("compute_buffer_mb", 800))
    rs      = float(auto.get("rs_buffer_mb", 63))
    kv128   = float(auto.get("kv_mb_at_128k", 500))
    safety  = float(auto.get("safety_margin_mb", 1024))

    kv = (ctx / 131072.0) * kv128
    budget = free_mb - kv - rs - compute - base - safety
    layers_on_gpu = max(0, min(total, int(budget // per)))
    ncmoe = total - layers_on_gpu
    if not quiet:
        print(f"  auto-ncmoe: {free_mb} MiB free, ctx={ctx} -> "
              f"{layers_on_gpu}/{total} expert layers on GPU, ncmoe={ncmoe}")
    return ncmoe


def _server_host(cfg, args=None):
    if getattr(args, "lan", False):
        return "0.0.0.0"
    if getattr(args, "host", None):
        return args.host
    return cfg.get("host", "127.0.0.1")


def _build_server_cmd(cfg, binary, model_path, port, ctx, host="127.0.0.1"):
    cmd = [
        binary, "-m", model_path,
        "-ngl", str(cfg.get("gpu_layers", 99)),
        "-c", str(ctx),
        "-fa", cfg.get("flash_attn", "on"),
        "-ctk", cfg.get("cache_k", "f16"),
        "-ctv", cfg.get("cache_v", "f16"),
        "--threads", str(cfg.get("threads", 4)),
        "-np", "1",
        "--host", host,
        "--port", str(port),
    ]
    mmproj = cfg.get("mmproj")
    if mmproj:
        mmproj_path = MODELS_DIR / mmproj
        if mmproj_path.exists():
            cmd += ["--mmproj", str(mmproj_path)]
        elif os.path.isfile(mmproj):
            cmd += ["--mmproj", mmproj]
        else:
            print(f"  warning: mmproj not found: {mmproj}", file=sys.stderr)

    # VRAM-aware -ncmoe (ported from run-qwen.ps1); only when auto_ncmoe is set.
    ncmoe = _compute_ncmoe(cfg, ctx)
    if ncmoe is not None:
        cmd += ["-ncmoe", str(ncmoe)]

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
    # Reads both the new bench format (decode_p50_tps / prefill_tps) and the
    # legacy format (gen_tps / prompt_tps).
    for filename in [f"bench-{key}.json", f"test-{key}.json"]:
        path = LOGS_DIR / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            data = data.get("results", [])
        if not isinstance(data, list):
            continue
        gens = [r.get("decode_p50_tps", r.get("gen_tps", 0))
                for r in data if isinstance(r, dict)]
        gens = [g for g in gens if g]
        prompts = [r.get("prefill_tps", r.get("prompt_tps", 0))
                   for r in data if isinstance(r, dict)]
        prompts = [p for p in prompts if p]
        if gens:
            avg_gen = round(sum(gens) / len(gens), 1)
            avg_prompt = round(sum(prompts) / len(prompts), 1) if prompts else None
            return avg_prompt, avg_gen
    return None, None


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_list(args):
    registry = load_registry()
    processes = _process_snapshots()
    external = _external_rows(processes)
    if not registry and not external:
        print("No models registered. Add one with:")
        print("  local-model add <path-to-gguf>")
        print("  local-model add hf:<huggingface-repo>")
        return

    headers = ["Model", "Source", "Kind", "Name", "Port", "Context", "Size", "RAM", "VRAM", "Status"]
    # Alignment for the first 9 (padded) columns; Status is last and unpadded
    # so its ANSI colour codes never throw off alignment.
    aligns = ["<", "<", "<", "<", ">", ">", ">", ">", ">"]

    rows = []
    for key, cfg in sorted(registry.items()):
        port = cfg.get("port", "?")
        ctx = cfg.get("context", "?")
        if isinstance(ctx, int):
            ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        else:
            ctx_str = str(ctx)

        if _is_remote(cfg):
            reachable = _check_endpoint(cfg)
            status = ("\033[36mremote ok\033[0m" if reachable
                      else "\033[31mremote down\033[0m")
            rows.append([key, "local-model", "llm", cfg.get("name", "?"),
                         "remote", ctx_str, "-", "-", "-", status])
            continue

        pid = get_running_pid(key)
        model_path = resolve_model_path(cfg)
        rss = processes.get(pid, {}).get("rss") if pid else None
        size = os.path.getsize(model_path) if model_path and os.path.isfile(model_path) else None
        if not model_path:
            status = "\033[31mmissing\033[0m"
        elif pid and check_health(port):
            status = f"\033[32mrunning\033[0m (:{port})"
        elif pid:
            status = "\033[33mstarting\033[0m"
        else:
            status = "\033[90mstopped\033[0m"

        rows.append([key, "local-model", "llm", cfg.get("name", "?"),
                     str(port), ctx_str, _fmt_bytes(size), _fmt_bytes(rss),
                     "-", status])

    for row in external:
        status = row["status"]
        if status == "running":
            status = f"\033[32mrunning\033[0m (:{row['port']})"
        elif status == "starting":
            status = f"\033[33mstarting\033[0m (:{row['port']})"
        elif status == "missing":
            status = "\033[31mmissing\033[0m"
        else:
            status = "\033[90mavailable\033[0m"
        rows.append([
            row["key"],
            row["source"],
            row["kind"],
            row["name"],
            str(row["port"]),
            str(row["context"]),
            _fmt_bytes(row.get("size")),
            _fmt_bytes(row.get("ram")),
            _fmt_bytes(row.get("vram")),
            status,
        ])

    # Column widths sized to the content (header + every cell), padded cols.
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows))
              for i in range(len(aligns))]

    def _fmt(cells):
        parts = [f"{cells[i]:{aligns[i]}{widths[i]}}" for i in range(len(aligns))]
        parts.append(cells[9])  # Status: last column, printed as-is
        return "  ".join(parts)

    header_line = _fmt(headers)
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(_fmt(r))


def cmd_start(args):
    _ensure_dirs()
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    if _is_remote(cfg):
        base = _model_endpoint(cfg)
        print(f"{cfg.get('name', key)} is a remote model (not started locally).")
        print(f"  endpoint: {base}")
        if _check_endpoint(cfg):
            print("  status:   reachable")
            print(f"\nPoint your client (Hermes / OpenClaw / any OpenAI SDK) at:")
            print(f"  {base}")
        else:
            print("  status:   NOT reachable", file=sys.stderr)
            print("  (is the remote host's model running and Tailscale up?)",
                  file=sys.stderr)
            sys.exit(1)
        return

    pid = get_running_pid(key)
    host = _server_host(cfg, args)

    if pid and check_health(cfg["port"]):
        print(f"{cfg['name']} is already running on port {cfg['port']} (PID {pid})")
        return

    port = cfg.get("port", 8080)

    if pid:
        try:
            _terminate_pid(pid)
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
    cmd = _build_server_cmd(cfg, binary, model_path, port, ctx, host)

    log_f = log_file_for(key)
    print(f"Starting {cfg['name']}...")
    print(f"  host: {host}  port: {port}  ctx: {ctx}  {_describe_config(cfg)}")

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
            if host == "0.0.0.0":
                lan_ip = _local_ip_address()
                if lan_ip:
                    print(f"  LAN: http://{lan_ip}:{port}/v1")
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
                _terminate_pid(pid)
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
        if _is_remote(cfg):
            found = True
            base = _model_endpoint(cfg)
            print(f"\n{cfg.get('name', key)}  [remote]")
            print(f"  Endpoint: {base}")
            print(f"  Health:   {'OK' if _check_endpoint(cfg) else 'UNREACHABLE'}")
            continue
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


def _monitor_items(registry, processes, gpu_per_pid):
    items = []
    for key, cfg in sorted(registry.items()):
        if _is_remote(cfg):
            continue
        pid = get_running_pid(key)
        if not pid:
            continue
        proc = processes.get(pid, {})
        items.append({
            "label": key,
            "source": "local-model",
            "pid": pid,
            "ram": proc.get("rss") or 0,
            "vram": (gpu_per_pid.get(pid) or {}).get("vram") or 0,
        })

    for row in _external_rows(processes, fast=True):
        if row.get("status") not in ("running", "starting"):
            continue
        pid = row.get("pid")
        items.append({
            "label": row["key"],
            "source": row["source"],
            "pid": pid,
            "ram": row.get("ram") or 0,
            "vram": row.get("vram") or ((gpu_per_pid.get(pid) or {}).get("vram") if pid else 0),
        })

    return items


_BAR_COLORS = ["31", "32", "33", "34", "35", "36", "91", "92", "93", "94", "95", "96"]


def _render_bar(title, total, used, segments, width=54):
    if not total:
        return [f"{title:<5} unavailable"]

    used = min(max(used or 0, 0), total)
    segs = [(label, max(0, value or 0), color)
            for label, value, color in segments if value and value > 0]
    known = min(sum(v for _, v, _ in segs), used)
    other = max(0, used - known)
    if other:
        segs.append(("other", other, "90"))

    cells = []
    used_cells = 0
    for i, (_label, value, color) in enumerate(segs):
        n = int(round(width * value / total))
        if value > 0 and n == 0:
            n = 1
        remaining = width - used_cells
        if i == len(segs) - 1:
            n = min(n, remaining)
        else:
            n = min(n, max(0, remaining))
        if n:
            cells.append(f"\033[{color}m" + ("#" * n) + "\033[0m")
            used_cells += n

    free_cells = max(0, width - used_cells)
    bar = "".join(cells) + "\033[90m" + ("-" * free_cells) + "\033[0m"
    pct = used / total * 100
    return [f"{title:<5} [{bar}] {pct:5.1f}%  {_fmt_bytes(used)} / {_fmt_bytes(total)}"]


def _monitor_frame():
    registry = load_registry()
    processes = _process_snapshots()
    items = _monitor_items(registry, processes, {})
    target_pids = {item.get("pid") for item in items if item.get("pid")}
    gpu_total, gpu_used, _gpu_free, gpu_per_pid = _gpu_memory(target_pids)
    ram_total, ram_used = _system_memory()

    for item in items:
        pid = item.get("pid")
        if pid and gpu_per_pid.get(pid):
            item["vram"] = gpu_per_pid[pid].get("vram") or item.get("vram") or 0

    color_by_label = {}
    for i, item in enumerate(items):
        color_by_label[item["label"]] = _BAR_COLORS[i % len(_BAR_COLORS)]

    ram_segments = [
        (item["label"], item.get("ram", 0), color_by_label[item["label"]])
        for item in items
    ]
    vram_segments = [
        (item["label"], item.get("vram", 0), color_by_label[item["label"]])
        for item in items
    ]
    known_ram = sum(item.get("ram", 0) or 0 for item in items)
    known_vram = sum(item.get("vram", 0) or 0 for item in items)
    other_ram = max(0, (ram_used or 0) - known_ram) if ram_used is not None else None
    other_vram = max(0, (gpu_used or 0) - known_vram) if gpu_used is not None else None

    lines = [
        "local-model monitor",
        time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
    ]
    lines.extend(_render_bar("RAM", ram_total, ram_used, ram_segments))
    lines.extend(_render_bar("VRAM", gpu_total, gpu_used, vram_segments))
    lines.append("")
    lines.append(f"{'Color':<7} {'Model':<32} {'Source':<12} {'PID':>7} {'RAM':>10} {'VRAM':>10}")
    lines.append("-" * 84)
    for item in items:
        color = color_by_label[item["label"]]
        swatch = f"\033[{color}m###\033[0m"
        pid = item.get("pid") or "-"
        lines.append(
            f"{swatch:<16} {item['label']:<32} {item['source']:<12} "
            f"{str(pid):>7} {_fmt_bytes(item.get('ram')):>10} {_fmt_bytes(item.get('vram')):>10}"
        )
    swatch = "\033[90m###\033[0m"
    lines.append(
        f"{swatch:<16} {'unattributed OS/process usage':<32} {'system':<12} "
        f"{'-':>7} {_fmt_bytes(other_ram):>10} {_fmt_bytes(other_vram):>10}"
    )
    return "\n".join(lines)


def _enable_virtual_terminal_output():
    if os.name != "nt" or not sys.stdout.isatty():
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def cmd_monitor(args):
    if args.once:
        print(_monitor_frame())
        return

    if not sys.stdout.isatty():
        try:
            while True:
                print(_monitor_frame())
                print()
                sys.stdout.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    _enable_virtual_terminal_output()
    try:
        sys.stdout.write("\033[?1049h\033[?25l\033[H\033[J")
        sys.stdout.flush()
        next_refresh = time.monotonic()
        while True:
            next_refresh += args.interval
            sys.stdout.write("\033[H")
            sys.stdout.write(_monitor_frame())
            sys.stdout.write("\033[J")
            sys.stdout.flush()
            time.sleep(max(0, next_refresh - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
    print("Stopped.")


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


def _chat_stream(port, prompt, max_tokens=512, model_name="bench",
                 enable_thinking=False):
    """Streaming chat completion with client-side TTFT measurement.

    Modern inference metrics: TTFT (time to first token) is timed client-side
    from the first streamed token; prefill/decode tok/s come from the server's
    own `timings` (authoritative), with a client-side fallback. Returns a dict
    with content, ttft, elapsed, prompt_tokens, tokens, prefill_tps,
    decode_tps, tpot_ms.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    content_parts, reasoning_parts = [], []
    usage, timings = {}, {}
    resp = urllib.request.urlopen(req, timeout=600)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {})
            c = delta.get("content") or ""
            rc = delta.get("reasoning_content") or ""
            if (c or rc) and ttft is None:
                ttft = time.perf_counter() - t0
            if c:
                content_parts.append(c)
            if rc:
                reasoning_parts.append(rc)
        if obj.get("usage"):
            usage = obj["usage"]
        if obj.get("timings"):
            timings = obj["timings"]
    elapsed = time.perf_counter() - t0

    content = "".join(content_parts)
    if not content and reasoning_parts:
        content = "".join(reasoning_parts)
    tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    prefill_tps = timings.get("prompt_per_second") or 0
    decode_tps = timings.get("predicted_per_second") or 0
    predicted_ms = timings.get("predicted_ms")

    if not decode_tps and ttft is not None and tokens > 1 and elapsed > ttft:
        decode_tps = (tokens - 1) / (elapsed - ttft)
    if predicted_ms and tokens >= 1:
        tpot_ms = predicted_ms / max(1, tokens)
    elif ttft is not None and tokens > 1 and elapsed > ttft:
        tpot_ms = (elapsed - ttft) * 1000.0 / (tokens - 1)
    else:
        tpot_ms = 0.0

    return {
        "content": content,
        "ttft": round(ttft, 3) if ttft is not None else None,
        "elapsed": round(elapsed, 2),
        "prompt_tokens": prompt_tokens,
        "tokens": tokens,
        "prefill_tps": round(prefill_tps, 1),
        "decode_tps": round(decode_tps, 1),
        "tpot_ms": round(tpot_ms, 1),
    }


def _bench_prompt(target_tokens):
    """Build a prompt of roughly target_tokens with a short generation task."""
    filler = ("The history of computing spans mechanical calculators, vacuum "
              "tubes, transistors, integrated circuits, and now accelerators "
              "for machine learning. Each era reshaped what software could do. ")
    target_chars = max(0, target_tokens - 40) * 4  # ~4 chars/token heuristic
    n = max(1, target_chars // len(filler))
    body = "\n".join(f"[{i}] {filler}" for i in range(n))
    return body + "\nSummarize the passage above in two sentences."


def _load_gsm8k(n):
    """Fetch + cache n GSM8K test rows via the HF datasets-server JSON API.

    No 'datasets' dependency needed - plain HTTP. Cached under ROOT/datasets.
    """
    cache = ROOT / "datasets" / "gsm8k_test.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if cache.exists():
        try:
            rows = json.loads(cache.read_text())
        except Exception:
            rows = []
    if len(rows) >= n:
        return rows[:n]
    rows = []
    offset = 0
    while len(rows) < n:
        length = min(100, n - len(rows))
        url = ("https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k"
               f"&config=main&split=test&offset={offset}&length={length}")
        try:
            r = urllib.request.urlopen(url, timeout=30)
            payload = json.loads(r.read())
        except Exception as e:
            print(f"  failed to fetch GSM8K: {e}", file=sys.stderr)
            break
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
    if rows:
        cache.write_text(json.dumps(rows, indent=2))
    return rows[:n]


def _extract_final_number(text):
    """Pull the final numeric answer from text (handles #### marker, $ and commas)."""
    import re
    has_marker = "####" in text
    tail = text.split("####")[-1] if has_marker else text
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", tail)
    if not nums:
        return None
    # After a #### marker the answer is the first number; otherwise fall back
    # to the last number in the text.
    pick = nums[0] if has_marker else nums[-1]
    s = pick.replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


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
                    _terminate_pid(pid)
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
            _terminate_pid(pid)
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
                _terminate_pid(pid)
            except ProcessLookupError:
                pass
            pid_file_for(key).unlink(missing_ok=True)
            print(f"\n{cfg['name']} stopped (was started for test).")


def _gpu_info():
    """GPU name / total VRAM / driver via nvidia-smi (empty dict if absent)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        parts = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        if len(parts) >= 3:
            return {"name": parts[0], "memory_total": parts[1], "driver": parts[2]}
    except Exception:
        pass
    return {}


def _engine_version(binary):
    """Version/build line from `<binary> --version` (best effort)."""
    if not binary or not os.path.isfile(binary):
        return "unknown"
    try:
        out = subprocess.run([binary, "--version"], capture_output=True,
                             text=True, timeout=30)
        text = (out.stderr or "") + "\n" + (out.stdout or "")
        ver = build = ""
        for line in text.splitlines():
            s = line.strip()
            if s.lower().startswith("version:"):
                ver = s
            elif s.lower().startswith("build"):
                build = s
        return " | ".join(x for x in (ver, build) if x) or "unknown"
    except Exception:
        return "unknown"


_QUANTS = ["IQ1_M", "IQ2_XXS", "IQ2_M", "IQ3_XXS", "IQ3_S", "IQ3_M",
           "IQ4_XS", "IQ4_NL", "Q2_K", "Q3_K", "Q4_K_M", "Q4_K_S",
           "Q5_K_M", "Q6_K", "Q8_0"]


def _provenance(cfg, ctx):
    """Capture engine/GPU/host/model/runtime so a result is reproducible and
    drift across engine, driver, or model upgrades is detectable."""
    binary = resolve_binary(cfg) or ""
    model_path = resolve_model_path(cfg) or cfg.get("file", "")
    size_gb = None
    quant = None
    try:
        if model_path and os.path.isfile(model_path):
            size_gb = round(os.path.getsize(model_path) / 1e9, 2)
    except OSError:
        pass
    base = os.path.basename(model_path or "")
    for q in _QUANTS:
        if q.lower() in base.lower():
            quant = q
            break
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": {"binary": binary, "version": _engine_version(binary)},
        "gpu": _gpu_info(),
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "model": {"file": model_path, "quant": quant, "size_gb": size_gb},
        "runtime": {
            "context": ctx,
            "cache_k": cfg.get("cache_k", "f16"),
            "cache_v": cfg.get("cache_v", "f16"),
            "flash_attn": cfg.get("flash_attn", "on"),
            "gpu_layers": cfg.get("gpu_layers", 99),
            "threads": cfg.get("threads", 4),
            "auto_ncmoe": cfg.get("auto_ncmoe"),  # calibration inputs (actual ncmoe is non-deterministic; printed live at start)
            "server_args": cfg.get("server_args", []),
        },
    }


def _print_provenance(meta):
    eng = meta.get("engine", {}).get("version", "?")
    gpu = meta.get("gpu", {})
    gpu_s = gpu.get("name", "?")
    if gpu.get("driver"):
        gpu_s += f" (drv {gpu['driver']})"
    quant = meta.get("model", {}).get("quant") or "?"
    print(f"  env: {eng}  |  {gpu_s}  |  quant {quant}")


def cmd_bench(args):
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    _ensure_clean_for_bench(registry, key)
    ctx_cap = args.ctx or cfg.get("context", 8192)
    port, started_by_us = _start_for_bench(registry, key, ctx_cap)

    actual_ctx = None
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=3)
        slots = json.loads(r.read())
        if slots:
            actual_ctx = slots[0].get("n_ctx")
    except Exception:
        pass
    ctx_limit = actual_ctx or ctx_cap

    # A couple of representative lengths (short / medium / longer), capped.
    candidates = [512, 8192, 32768]
    test_ctx = [c for c in candidates if c <= ctx_limit] or [min(candidates)]
    iters = getattr(args, "iters", None) or 3

    def pct(xs, p):
        xs = sorted(x for x in xs if x is not None)
        if not xs:
            return 0.0
        i = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
        return xs[i]

    print(f"\nBenchmarking {cfg['name']} on port {port} (ctx_limit={ctx_limit})")
    print(f"  {iters} iters/length, 1 warmup discarded, streaming TTFT\n")
    hdr = (f"{'context':>8}  {'prompt':>7}  {'TTFT p50':>9}  {'TTFT p90':>9}  "
           f"{'decode p50':>11}  {'decode p90':>11}  {'prefill':>8}")
    print(hdr)
    print("-" * len(hdr))

    import uuid
    results = []
    for target in test_ctx:
        base = _bench_prompt(target)
        # Unique prefix per request busts llama.cpp prompt-prefix caching so
        # every measured request pays a true (cold) prefill -> honest TTFT.
        def _fresh():
            return f"[run {uuid.uuid4().hex[:8]}] {base}"
        try:
            _chat_stream(port, _fresh(), max_tokens=128)  # warmup (discarded)
        except Exception:
            pass
        samples = []
        for _ in range(iters):
            try:
                samples.append(_chat_stream(port, _fresh(), max_tokens=128))
            except Exception as e:
                print(f"{target:>8}  ERROR: {e}")
        if not samples:
            continue
        ttfts = [s["ttft"] for s in samples]
        decodes = [s["decode_tps"] for s in samples]
        prefills = [s["prefill_tps"] for s in samples if s["prefill_tps"]]
        ptoks = samples[0]["prompt_tokens"]
        row = {
            "context": target,
            "prompt_tokens": ptoks,
            "ttft_p50_s": round(pct(ttfts, 50), 2),
            "ttft_p90_s": round(pct(ttfts, 90), 2),
            "decode_p50_tps": round(pct(decodes, 50), 1),
            "decode_p90_tps": round(pct(decodes, 90), 1),
            "prefill_tps": round(sum(prefills) / len(prefills), 1) if prefills else 0.0,
        }
        results.append(row)
        print(f"{target:>8}  {ptoks:>7}  {row['ttft_p50_s']:>8.2f}s  "
              f"{row['ttft_p90_s']:>8.2f}s  {row['decode_p50_tps']:>11.1f}  "
              f"{row['decode_p90_tps']:>11.1f}  {row['prefill_tps']:>8.0f}")

    if results:
        print(f"\nBest median decode: {max(r['decode_p50_tps'] for r in results):.1f} tok/s")

    meta = _provenance(cfg, ctx_limit)
    _print_provenance(meta)
    out = LOGS_DIR / f"bench-{key}.json"
    out.write_text(json.dumps({"meta": meta, "results": results},
                              indent=2, default=str) + "\n")
    print(f"Results saved to {out}")

    if started_by_us:
        pid = get_running_pid(key)
        if pid:
            _terminate_pid(pid)
            pid_file_for(key).unlink(missing_ok=True)
            print(f"\n{cfg['name']} stopped (was started for benchmark).")


def cmd_eval(args):
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    _ensure_clean_for_bench(registry, key)
    port, started_by_us = _start_for_bench(registry, key)

    n = getattr(args, "questions", None) or 20
    print(f"\nEvaluating {cfg['name']} on port {port}")

    # ---- GSM8K: reasoning accuracy (exact-match auto-scored) ----
    print(f"\n== GSM8K reasoning ({n} questions, exact-match auto-scored) ==")
    questions = _load_gsm8k(n)
    gsm_acc = None
    if not questions:
        print("  could not load GSM8K (offline?); skipping")
    else:
        correct = 0
        rates = []
        for i, q in enumerate(questions, 1):
            gold = _extract_final_number(q.get("answer", ""))
            prompt = (q.get("question", "") + "\n\nSolve step by step. End with "
                      "the final numeric answer on its own line after '#### '.")
            try:
                r = _chat_stream(port, prompt, max_tokens=2048, enable_thinking=True)
            except Exception as e:
                print(f"  Q{i:>2}: ERROR {e}")
                continue
            pred = _extract_final_number(r["content"])
            ok = gold is not None and pred is not None and abs(pred - gold) < 1e-4
            correct += 1 if ok else 0
            rates.append(r["decode_tps"])
            print(f"  Q{i:>2}: {'PASS' if ok else 'fail'}  gold={gold}  "
                  f"pred={pred}  ({r['decode_tps']:.0f} tok/s)")
        gsm_acc = round(100.0 * correct / len(questions), 1)
        avg = sum(rates) / len(rates) if rates else 0
        print(f"  -> GSM8K: {correct}/{len(questions)} = {gsm_acc}%   "
              f"avg decode {avg:.1f} tok/s")

    # ---- Our own needle retrieval (auto-scored PASS/FAIL) ----
    print(f"\n== Needle retrieval (our own test, auto-scored) ==")
    needle = []
    for ctx_tokens in [2000, 16000]:
        hay = _build_haystack(ctx_tokens)
        prompt = (f"{hay}\n\nWhat is the secret project codename? "
                  "Answer with just the codename.")
        try:
            r = _chat_stream(port, prompt, max_tokens=64, enable_thinking=False)
            found = "midnight falcon" in r["content"].lower()
            print(f"  ~{ctx_tokens:>6} tok: {'PASS' if found else 'fail'}  "
                  f"(prompt={r['prompt_tokens']} tok, {r['decode_tps']:.0f} tok/s)")
            needle.append({"ctx": ctx_tokens, "found": found,
                           "prompt_tokens": r["prompt_tokens"]})
        except Exception as e:
            print(f"  ~{ctx_tokens} tok: ERROR {e}")

    meta = _provenance(cfg, cfg.get("context", 8192))
    _print_provenance(meta)
    out = LOGS_DIR / f"eval-{key}.json"
    out.write_text(json.dumps({"meta": meta, "gsm8k_accuracy_pct": gsm_acc,
                               "needle": needle}, indent=2, default=str) + "\n")
    print(f"\nResults saved to {out}")

    if started_by_us:
        pid = get_running_pid(key)
        if pid:
            _terminate_pid(pid)
            pid_file_for(key).unlink(missing_ok=True)
            print(f"\n{cfg['name']} stopped (was started for eval).")


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
        linked = dest.exists()
        if not linked:
            try:
                os.symlink(src_path, dest)
                print(f"Linked {src_path.name} -> {dest}")
                linked = True
            except (OSError, NotImplementedError):
                print("Symlink unavailable (needs admin or Developer Mode on "
                      "Windows); registering absolute path instead.")
        if not name:
            name = src_path.stem.lower().replace(" ", "-")
        source = str(dest) if linked else str(src_path)

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
        "file": source,
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

def _coerce_value(v):
    """Best-effort type coercion for `edit --set key=value` values."""
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v[:1] in "[{":
        try:
            return json.loads(v)
        except Exception:
            pass
    return v


def _pi_installed():
    """True if pi appears installed on this machine: the `pi` binary is on PATH,
    or the ~/.pi/agent config directory exists. Lets sync be a clean no-op on
    machines without pi rather than emitting confusing output."""
    if shutil.which("pi"):
        return True
    return (Path(os.path.expanduser("~")) / ".pi" / "agent").is_dir()


def cmd_sync_pi(args):
    # Pi is registry-driven: the extension reads ~/.local-model/registry.json
    # directly, so there are no files to patch. This just guides you to refresh.
    if not _pi_installed():
        print("pi not detected on this machine; nothing to do.")
        return
    print("pi reads the local-model registry directly -- no file sync needed.")
    print("After adding or editing models, run /reload in pi (or restart it)")
    print("to refresh the model list, context windows, and ports.")


def cmd_edit(args):
    registry = load_registry()
    if not registry:
        print("No models registered.", file=sys.stderr)
        sys.exit(1)
    key = get_model_key(registry, args.model)
    if key not in registry:
        print(f"Unknown model '{args.model}'. Run 'local-model list'.", file=sys.stderr)
        sys.exit(1)
    cfg = registry[key]

    changes = []

    def _set(field, value, label=None):
        old = cfg.get(field, "(unset)")
        cfg[field] = value
        changes.append(f"{label or field}: {old!r} -> {value!r}")

    if getattr(args, "rename_key", None) and args.rename_key != key:
        new_key = args.rename_key
        if new_key in registry:
            print(f"  cannot rename: key '{new_key}' already exists", file=sys.stderr)
            sys.exit(1)
        registry[new_key] = registry.pop(key)
        cfg = registry[new_key]
        # Move the per-key sidecar files (pid, log, bench/test/eval results).
        for old_f, new_f in [
            (pid_file_for(key), pid_file_for(new_key)),
            (log_file_for(key), log_file_for(new_key)),
            (LOGS_DIR / f"bench-{key}.json", LOGS_DIR / f"bench-{new_key}.json"),
            (LOGS_DIR / f"test-{key}.json", LOGS_DIR / f"test-{new_key}.json"),
            (LOGS_DIR / f"eval-{key}.json", LOGS_DIR / f"eval-{new_key}.json"),
        ]:
            try:
                if old_f.exists():
                    old_f.replace(new_f)
            except OSError:
                pass
        changes.append(f"key: '{key}' -> '{new_key}'")
        key = new_key

    if args.name is not None:
        _set("name", args.name)
    if args.description is not None:
        _set("notes", args.description, "description")
    if args.context is not None:
        _set("context", args.context)
    if args.cache_k is not None:
        _set("cache_k", args.cache_k)
    if args.cache_v is not None:
        _set("cache_v", args.cache_v)
    if args.threads is not None:
        _set("threads", args.threads)
    if args.flash_attn is not None:
        _set("flash_attn", args.flash_attn)
    if args.gpu_layers is not None:
        _set("gpu_layers", args.gpu_layers)
    if args.server_args is not None:
        import shlex
        _set("server_args", shlex.split(args.server_args))
    if args.port is not None:
        clash = [k for k, c in registry.items()
                 if k != key and c.get("port") == args.port]
        if clash:
            print(f"  warning: port {args.port} also used by: {', '.join(clash)}",
                  file=sys.stderr)
        _set("port", args.port)
    if args.set:
        for kv in args.set:
            if "=" not in kv:
                print(f"  skipping malformed --set '{kv}' (need KEY=VALUE)",
                      file=sys.stderr)
                continue
            k, _, v = kv.partition("=")
            _set(k.strip(), _coerce_value(v))

    if not changes:
        # No edits requested -> act as an inspector and show current config.
        print(f"\n{cfg.get('name', key)}  [{key}]")
        for k in sorted(cfg):
            val = cfg[k]
            if isinstance(val, (dict, list)):
                val = json.dumps(val)
            print(f"  {k:<16} {val}")
        print("\nEdit examples:")
        print(f"  local-model edit {key} --port 8090 --context 131072")
        print("  local-model edit " + key + ' --server-args "--no-mmap --jinja"')
        print(f"  local-model edit {key} --set threads=12")
        return

    save_registry(registry)
    print(f"Updated '{key}':")
    for c in changes:
        print(f"  {c}")

    # Pi reads the registry directly (registry-driven extension); remind to
    # reload when a pi-visible field changed.
    if _pi_installed() and (args.name is not None or args.context is not None
                            or args.port is not None
                            or getattr(args, "rename_key", None)):
        print("  (pi reads the registry live; run /reload in pi to apply.)")

    if get_running_pid(key):
        print(f"\nNote: {cfg.get('name', key)} is running; changes take effect on "
              f"next start  (local-model stop {key} ; local-model start {key}).")


def cmd_add_remote(args):
    registry = load_registry()
    url = _normalize_remote_url(args.url)
    name = args.name or "remote model"
    key = _unique_key(registry, _safe_key(args.name or "remote"))
    registry[key] = {
        "name": name,
        "remote": True,
        "url": url,
        "context": args.context or 8192,
        "notes": f"Remote model at {url}",
    }
    save_registry(registry)
    print(f"Registered remote model '{key}':")
    print(f"  name:    {name}")
    print(f"  url:     {url}")
    print(f"  context: {registry[key]['context']}")
    reachable = _check_endpoint(registry[key])
    print(f"  reachable: {'yes' if reachable else 'no (remote down or Tailscale off)'}")
    print(f"\nVerify with: local-model start {key}")
    print(f"Use it by pointing your client at: {url}")


def cmd_scan(args):
    try:
        ports = _parse_port_list(args.ports)
        hosts = _expand_scan_targets(args.target or _default_scan_targets())
    except ValueError as exc:
        print(f"scan: {exc}", file=sys.stderr)
        sys.exit(1)

    if not hosts:
        print("scan: no hosts to scan", file=sys.stderr)
        sys.exit(1)
    if len(hosts) > args.max_hosts:
        print(f"scan: refusing to scan {len(hosts)} hosts (limit {args.max_hosts})",
              file=sys.stderr)
        print("Use --max-hosts N or a narrower --target.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(hosts)} host(s) x {len(ports)} port(s) for /v1/models...")

    found = []
    workers = max(1, min(args.workers, len(hosts) * len(ports)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_probe_openai_models, host, port, args.timeout)
            for host in hosts
            for port in ports
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                item = future.result()
            except Exception:
                item = None
            if item:
                found.append(item)

    def _sort_key(item):
        try:
            host_key = ipaddress.ip_address(item["host"])
        except ValueError:
            host_key = item["host"]
        return str(host_key), item["port"]

    found.sort(key=_sort_key)
    if not found:
        print("No OpenAI-compatible model endpoints found.")
        return

    registry = load_registry()
    existing_urls = {
        _normalize_remote_url(cfg.get("url", ""))
        for cfg in registry.values()
        if _is_remote(cfg) and cfg.get("url")
    }
    registered = []

    print(f"\n{'Endpoint':<28} {'Models'}")
    print("-" * 80)
    for item in found:
        models = ", ".join(item["models"][:4])
        if len(item["models"]) > 4:
            models += f", +{len(item['models']) - 4} more"
        print(f"{item['url']:<28} {models}")

        if not args.register:
            continue

        url = _normalize_remote_url(item["url"])
        if url in existing_urls:
            registered.append((url, "already registered"))
            continue
        model_name = item["models"][0] if item["models"] else f"{item['host']}:{item['port']}"
        key = _unique_key(registry, _safe_key(model_name))
        registry[key] = {
            "name": model_name,
            "remote": True,
            "url": url,
            "context": args.context,
            "models": item["models"],
            "notes": f"Discovered by network scan at {item['host']}:{item['port']}",
        }
        existing_urls.add(url)
        registered.append((url, key))

    if args.register:
        if any(key != "already registered" for _url, key in registered):
            save_registry(registry)
        print("\nRegistration:")
        for url, key in registered:
            if key == "already registered":
                print(f"  {url}: already registered")
            else:
                print(f"  {url}: registered as '{key}'")
    else:
        print("\nRegister discovered endpoints with: local-model scan --register")


def _tailscale_bin():
    """Locate the tailscale CLI (PATH, or the default Windows install path)."""
    found = shutil.which("tailscale")
    if found:
        return found
    if os.name == "nt":
        candidate = r"C:\Program Files\Tailscale\tailscale.exe"
        if os.path.isfile(candidate):
            return candidate
    return None


def _parse_serve_url(text):
    """Pull the https://<host>.ts.net URL out of `tailscale serve` output."""
    m = re.search(r"https://\S+\.ts\.net\S*", text or "")
    return m.group(0) if m else None


def _tailscale_https_url(ts):
    """Best-effort https URL from `tailscale status --json` (Self.DNSName)."""
    try:
        r = subprocess.run([ts, "status", "--json"],
                           capture_output=True, text=True, timeout=10)
        data = json.loads(r.stdout)
        dns = (data.get("Self") or {}).get("DNSName", "").rstrip(".")
        return f"https://{dns}" if dns else None
    except Exception:
        return None


def cmd_serve(args):
    """Expose a local model over Tailscale HTTPS and/or the LAN."""
    _ensure_dirs()
    registry = load_registry()
    key = get_model_key(registry, args.model)
    cfg = get_model(registry, args.model)

    if _is_remote(cfg):
        print(f"'{key}' is a remote model; serve is for local models only.",
              file=sys.stderr)
        sys.exit(1)

    lan = getattr(args, "lan", False)
    port = cfg.get("port", 8080)
    ts = _tailscale_bin()
    if not ts and not lan:
        print("tailscale CLI not found.", file=sys.stderr)
        print("Install Tailscale and sign in, then retry.", file=sys.stderr)
        sys.exit(1)

    # --off: remove the proxy mapping and exit.
    if getattr(args, "off", False):
        if not ts:
            print("tailscale CLI not found.", file=sys.stderr)
            sys.exit(1)
        r = subprocess.run([ts, "serve", "--https=443", "off"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print("Stopped serving (Tailscale https/443 mapping removed).")
        else:
            print((r.stderr or r.stdout or "failed to stop serve").strip(),
                  file=sys.stderr)
            sys.exit(1)
        return

    # 1. Ensure the model is up. For LAN serving, restart a localhost-bound
    #    process so llama-server listens on 0.0.0.0.
    pid = get_running_pid(key)
    if lan and pid and check_health(port) and not _check_lan_endpoint(port):
        print(f"{cfg.get('name', key)} is running on localhost; restarting for LAN...")
        _terminate_pid(pid)
        time.sleep(2)
        pid = None

    if pid and check_health(port):
        print(f"{cfg.get('name', key)} already running on port {port} (PID {pid}).")
    else:
        cmd_start(args)

    # 2. Map the device's :443 to the local model port over Tailscale when
    #    available. --lan can be used on machines without Tailscale installed.
    url = None
    if ts:
        print(f"\nExposing port {port} over Tailscale (HTTPS)...")
        r = subprocess.run(
            [ts, "serve", "--bg", "--https=443", f"http://127.0.0.1:{port}"],
            capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            if "not enabled" in out.lower():
                print("Tailscale Serve is not enabled on your tailnet.", file=sys.stderr)
                print("Enable HTTPS Certificates in the admin console, then retry:",
                      file=sys.stderr)
                print("  https://login.tailscale.com/admin/dns", file=sys.stderr)
            else:
                print(out.strip() or "tailscale serve failed", file=sys.stderr)
            if not lan:
                sys.exit(1)
        else:
            url = _parse_serve_url(out) or _tailscale_https_url(ts)

    # 3. Print client-ready details.
    base = (url.rstrip("/") + "/v1") if url else None
    lan_base = _lan_model_url(port) if lan else None
    mp = cfg.get("file") or cfg.get("model")
    model_id = os.path.basename(str(mp)) if mp else key

    if lan:
        print("\nServing on LAN:")
        if lan_base:
            print(f"  OpenAI: {lan_base}")
            print(f"  Scan:   local-model scan --target {lan_base.split('//', 1)[1].split(':', 1)[0]} --ports {port}")
        else:
            print("  LAN IP could not be detected; use your machine's local IP.")
        print("  Note:   allow inbound TCP for this port in the OS firewall.")

    if url:
        print("\nServing over Tailscale:")
    elif not lan:
        print("Serving over Tailscale:")
    if url:
        print(f"  URL:    {url.rstrip('/')}")
        print(f"  OpenAI: {base}")
    print(f"  Model:  {cfg.get('name', key)} (port {port})")
    print(f"  Health: {'OK' if check_health(port) else 'NOT READY'}")
    client_base = base or lan_base
    if client_base:
        print("\nPoint a remote client (Hermes / OpenClaw / any OpenAI SDK) at:")
        print(f"  base_url = {client_base}")
        print(f"  model    = {model_id}")
        print("  api_key  = not-needed")
    if url:
        print(f"\nStop Tailscale serving with: local-model serve {key} --off")
    if lan:
        print(f"Stop the LAN model server with: local-model stop {key}")


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
    print(f"  {'serve <model> [--lan]':<30} Start + expose a model over Tailscale/LAN")
    print(f"  {'stop <model|all>':<30} Stop a running model server")
    print(f"  {'status':<30} Show running servers with health info")
    print(f"  {'monitor [--once]':<30} Show RAM/VRAM attribution bars")
    print(f"  {'scan [--register]':<30} Scan the LAN for OpenAI-compatible models")
    print(f"  {'test <model> [--prompts N]':<30} Run quality tests (reasoning, coding, factual)")
    print(f"  {'bench <model> [--ctx N]':<30} Run speed benchmark at multiple context sizes")
    print(f"  {'eval <model> [--questions N]':<30} Accuracy eval (GSM8K reasoning + needle)")
    print(f"  {'add <path|hf:repo> [name]':<30} Register a new GGUF model")
    print(f"  {'info <model>':<30} Show model details (size, config, GGUF metadata)")
    print(f"  {'edit <model> [--port N ...]':<30} Edit a model's name, port, context, runtime args")
    print(f"  {'add-remote <url> [name]':<30} Register a remote model (e.g. over Tailscale)")
    print(f"  {'sync-pi':<30} How to refresh pi after registry changes")
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
              local-model serve bonsai --lan                Expose on LAN + print scan target
              local-model stop all                          Stop all running servers
              local-model monitor --once                     Show RAM/VRAM attribution bars
              local-model scan --register                    Find and register LAN model servers
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

    p = sub.add_parser("serve",
                       help="Start (if needed) and expose a model over Tailscale/LAN")
    p.add_argument("model", help="Model name")
    p.add_argument("--ctx", type=int, help="Override context window when starting")
    p.add_argument("--lan", action="store_true",
                   help="Bind to 0.0.0.0 and print the LAN OpenAI base URL")
    p.add_argument("--off", action="store_true",
                   help="Stop serving (remove the Tailscale HTTPS mapping)")

    sub.add_parser("status", help="Show running servers with health info")

    p = sub.add_parser("monitor", help="Show RAM/VRAM attribution bars")
    p.add_argument("--interval", type=float, default=2.0, help="Refresh interval seconds")
    p.add_argument("--once", action="store_true", help="Print one frame and exit")

    p = sub.add_parser("scan", help="Scan the LAN for OpenAI-compatible model servers")
    p.add_argument("--target", action="append",
                   help="Host, IP, or CIDR to scan (repeatable; default localhost + LAN /24)")
    p.add_argument("--ports", default="8080,8000,11434,1234,5000,5001,8880,8808",
                   help="Comma-separated ports/ranges to scan")
    p.add_argument("--timeout", type=float, default=0.35,
                   help="HTTP timeout per endpoint in seconds")
    p.add_argument("--workers", type=int, default=64,
                   help="Concurrent probes")
    p.add_argument("--max-hosts", type=int, default=512,
                   help="Safety limit for expanded targets")
    p.add_argument("--register", action="store_true",
                   help="Register discovered endpoints as remote models")
    p.add_argument("--context", type=int, default=8192,
                   help="Context window used for registrations")

    p = sub.add_parser("test", help="Run quality tests against a running model")
    p.add_argument("model", help="Model name")
    p.add_argument("--prompts", type=int, help="Number of test prompts to run")

    p = sub.add_parser("bench", help="Run speed benchmark (TTFT + decode tok/s)")
    p.add_argument("model", help="Model name")
    p.add_argument("--ctx", type=int, help="Max context to test")
    p.add_argument("--iters", type=int, help="Iterations per context length (default 3)")

    p = sub.add_parser("eval", help="Accuracy eval (GSM8K reasoning + needle retrieval)")
    p.add_argument("model", help="Model name")
    p.add_argument("--questions", type=int, help="GSM8K questions to run (default 20)")

    p = sub.add_parser("add", help="Register a new GGUF model")
    p.add_argument("source", help="Path to GGUF file or hf:<repo> for Hugging Face")
    p.add_argument("name", nargs="?", help="Short name for the model")

    p = sub.add_parser("add-remote", help="Register a remote (already-running) model by URL")
    p.add_argument("url", help="Base URL or host:port of the remote llama-server")
    p.add_argument("name", nargs="?", help="Short name for the model")
    p.add_argument("--context", type=int, help="Context window (default 8192)")

    p = sub.add_parser("info", help="Show model details")
    p.add_argument("model", help="Model name")

    p = sub.add_parser("edit", help="Edit a registered model's settings")
    p.add_argument("model", help="Model name/key")
    p.add_argument("--name", help="Display name")
    p.add_argument("--rename-key", metavar="NEWKEY",
                   help="Rename the registry key/identifier used in commands")
    p.add_argument("--description", help="Description / notes")
    p.add_argument("--port", type=int, help="Port")
    p.add_argument("--context", type=int, help="Context window size")
    p.add_argument("--cache-k", help="KV cache K type (f16, q8_0, turbo3, ...)")
    p.add_argument("--cache-v", help="KV cache V type")
    p.add_argument("--threads", type=int, help="CPU threads")
    p.add_argument("--flash-attn", choices=["on", "off", "auto"], help="Flash attention")
    p.add_argument("--gpu-layers", type=int, help="GPU layers (-ngl)")
    p.add_argument("--server-args", help="Raw extra server args, quoted; replaces existing")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="Set an arbitrary config field (repeatable; value auto-typed)")

    p = sub.add_parser("sync-pi", help="How to refresh pi after registry changes (registry-driven)")
    p.add_argument("model", nargs="?", help="Model name (default: all registered)")

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
        "serve": cmd_serve,
        "stop": cmd_stop,
        "status": cmd_status,
        "monitor": cmd_monitor,
        "scan": cmd_scan,
        "test": cmd_test,
        "bench": cmd_bench,
        "eval": cmd_eval,
        "add": cmd_add,
        "add-remote": cmd_add_remote,
        "info": cmd_info,
        "edit": cmd_edit,
        "sync-pi": cmd_sync_pi,
        "config": cmd_config,
        "help": cmd_help,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
