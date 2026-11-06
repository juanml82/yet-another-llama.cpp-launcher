#!/usr/bin/env python3
"""launcher.py — Llama.cpp / ik_llama launcher (backend FastAPI)."""

import argparse
import json
import os
import re
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gguf_read import read_gguf_meta

# ── Persistent config ───────────────────────────────────────────
CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "model_dirs": [
        "/mnt/shared/LM_Studio_Models",
        "/mnt/shared/ComfyUI/models/text_encoders",
    ],
    "backends": {
        "llamacpp": {"bin": "/home/juan/llama.cpp/build/bin",    "port": 8502},
        "ik_llama": {"bin": "/home/juan/ik_llama.cpp/build/bin", "port": 8507},
    },
    "launch_options": {
        "llamacpp": {
            "moe_cache_mib": 0,
        },
        "ik_llama": {
            "jinja": False,
        },
    },
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception as e:
            print(f"[config] error reading {CONFIG_FILE}: {e}")
    return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


_config = load_config()
_model_dirs = list(_config["model_dirs"])

PRESETS_FILE = Path(__file__).parent / "presets.json"
HTML_FILE    = Path(__file__).parent / "index.html"

# ── Manual path overrides (survive rescans) ─────────────────────
OVERRIDES_FILE = Path(__file__).parent / "overrides.json"


def load_overrides() -> dict:
    """
    Returns {model_name: {"mtp_path": "...", "mmproj_path": "...", ...}}
    """
    if OVERRIDES_FILE.exists():
        try:
            return json.loads(OVERRIDES_FILE.read_text())
        except Exception:
            pass
    return {}


def save_overrides(ov: dict):
    OVERRIDES_FILE.write_text(json.dumps(ov, indent=2))


_overrides: dict = load_overrides()

# ── Regex helpers ───────────────────────────────────────────────
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})")

# A companion is recognised by anchored prefix or anchored suffix
_COMPANION_PREFIX = re.compile(r'^(mmproj|mtp|dflash)[-_.]', re.IGNORECASE)
_COMPANION_SUFFIX = re.compile(
    r'[-_.](mmproj|mtp|dflash)'
    r'(?:[-_.](?:q|iq)\d+[_a-z0-9]*|[-_.](?:f16|f32|bf16|fp8|fp16|int8|int4))?'
    r'\.gguf$',
    re.IGNORECASE,
)


def _companion_kind(fname: str) -> Optional[str]:
    m = _COMPANION_PREFIX.match(fname)
    if m:
        return m.group(1).lower()
    m = _COMPANION_SUFFIX.search(fname)
    if m:
        return m.group(1).lower()
    return None


SIZE_RE = re.compile(r"(?:(\d+)x)?\s*(\d+(?:\.\d+)?)\s*(?:b|B)\b")

QUANT_RE = re.compile(
    r"[-._]?(?:"
    r"((?:q|iq)\d+[_a-z0-9]*)"
    r"|(f16|f32|bf16|fp8|fp16)"
    r"|(int8|int4)"
    r"|qat"
    r")$",
    re.IGNORECASE,
)


def iec(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PiB"


def detect_size(fname: str) -> Optional[str]:
    s = fname.lower().replace("_", "-")
    m = SIZE_RE.search(s)
    if m:
        mult, num = m.groups()
        return f"{mult}x{num}b" if mult else f"{num}b"
    return None


def detect_quant(fname: str) -> str:
    s = fname.lower().rsplit(".", 1)[0]
    s = SHARD_RE.sub("", s)
    m = QUANT_RE.search(s)
    if m:
        raw = next(g for g in m.groups() if g)
        return raw.upper()
    return ""


def _normalize_key(fname: str) -> str:
    s = fname.lower()
    if "." in s:
        s = s.rsplit(".", 1)[0]
    s = SHARD_RE.sub("", s)
    for _ in range(6):
        orig = s
        s = re.sub(r"[-._]?(?:q|iq)\d+[_a-z0-9]*$", "", s)
        s = re.sub(r"[-._]?(f16|f32|bf16|fp8|int8|int4|qat)$", "", s)
        s = re.sub(r"^(?:mmproj|mtp|dflash)[-._]?", "", s)
        s = re.sub(r"[-._]?mmproj.*$", "", s)
        s = re.sub(r"[-._]?mtp$", "", s)
        s = re.sub(r"[-._]?dflash$", "", s)
        s = re.sub(r"[-._]?draft.*$", "", s)
        if s == orig:
            break
    return s or fname.lower().rsplit(".", 1)[0]


# ── Model record ────────────────────────────────────────────────
@dataclass
class Model:
    name: str
    size: str
    multimodal: bool
    mtp: bool
    dflash: bool
    quant: str
    file_size: int
    path: str
    mmproj_path: str
    mmproj_paths: list
    mtp_path: str
    mtp_paths: list
    dflash_path: str
    dflash_paths: list
    source_dir: str
    is_sharded: bool
    shard_count: int
    arch: str = ""
    is_moe: bool = False
    n_expert: int = 0
    n_expert_used: int = 0
    n_layer: int = 0
    context_length: int = 0
    has_embedded_mtp: bool = False
    has_swa: bool = False
    swa_window: int = 0


# ── Scanner ─────────────────────────────────────────────────────
def scan_models(directories: list) -> list:
    models: list = []

    for base in directories:
        base_p = Path(base)
        if not base_p.exists():
            continue

        # onerror: an unreadable subdirectory won't abort the whole walk
        try:
            walker = os.walk(base, onerror=lambda e: print(f"[scan] {e}"))
            roots = list(walker)
        except OSError as e:
            print(f"[scan] skipping {base}: {e}")
            continue

        for root, _dirs, files in roots:
            # Only accessible .gguf files (avoids broken HF symlinks)
            gguf_files: list = []
            for f in files:
                if not f.lower().endswith(".gguf"):
                    continue
                full = os.path.join(root, f)
                try:
                    if os.path.isfile(full):
                        gguf_files.append(f)
                except OSError:
                    continue

            if not gguf_files:
                continue

            mmproj_files: list = []
            mtp_files: list = []
            dflash_files: list = []
            main_files: list = []

            for f in gguf_files:
                kind = _companion_kind(f)
                if kind == "mmproj":
                    mmproj_files.append(f)
                elif kind == "mtp":
                    mtp_files.append(f)
                elif kind == "dflash":
                    dflash_files.append(f)
                else:
                    main_files.append(f)

            if not main_files:
                continue

            groups: dict = {}
            for f in main_files:
                groups.setdefault(_normalize_key(f), []).append(f)

            for key, fgroup in groups.items():
                shards = [f for f in fgroup if SHARD_RE.search(f)]
                if shards:
                    launch_file = sorted(
                        shards,
                        key=lambda f: int(SHARD_RE.search(f).group(1))
                    )[0]
                    shard_count = len(shards)
                else:
                    launch_file = fgroup[0]
                    shard_count = 0

                launch_path = os.path.join(root, launch_file)
                if not os.path.isfile(launch_path):
                    continue

                # Sum sizes, tolerant of files that disappear
                total_size = 0
                for f in fgroup:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

                try:
                    meta = read_gguf_meta(launch_path)
                except Exception as e:
                    print(f"[scan] error parsing {launch_file}: {e}")
                    meta = None

                if meta and meta.file_type_name:
                    quant = meta.file_type_name
                else:
                    quant = detect_quant(launch_file)

                models.append(Model(
                    name=key,
                    size=detect_size(launch_file) or "",
                    multimodal=bool(mmproj_files),
                    mtp=bool(mtp_files),
                    dflash=bool(dflash_files),
                    quant=quant,
                    file_size=total_size,
                    path=launch_path,
                    mmproj_path=os.path.join(root, mmproj_files[0]) if mmproj_files else "",
                    mmproj_paths=[os.path.join(root, f) for f in mmproj_files],
                    mtp_path=os.path.join(root, mtp_files[0]) if mtp_files else "",
                    mtp_paths=[os.path.join(root, f) for f in mtp_files],
                    dflash_path=os.path.join(root, dflash_files[0]) if dflash_files else "",
                    dflash_paths=[os.path.join(root, f) for f in dflash_files],
                    source_dir=root,
                    is_sharded=shard_count > 0,
                    shard_count=shard_count,
                    arch=meta.arch if meta else "",
                    is_moe=meta.is_moe if meta else False,
                    n_expert=meta.n_expert if meta else 0,
                    n_expert_used=meta.n_expert_used if meta else 0,
                    n_layer=meta.n_layer if meta else 0,
                    context_length=meta.context_length if meta else 0,
                    has_embedded_mtp=meta.has_embedded_mtp if meta else False,
                    has_swa=meta.has_swa if meta else False,
                    swa_window=meta.swa_window if meta else 0,
                ))

    # Apply manual overrides after the scan so they survive rescans
    for m in models:
        ov = _overrides.get(m.name, {})
        if ov.get("mtp_path"):
            m.mtp_path = ov["mtp_path"]
        if ov.get("mmproj_path"):
            m.mmproj_path = ov["mmproj_path"]

    return models


# ── Presets ─────────────────────────────────────────────────────
class PresetIn(BaseModel):
    name: str
    glob: str
    params: dict


def load_presets() -> list:
    if PRESETS_FILE.exists():
        return json.loads(PRESETS_FILE.read_text())
    defaults = [
        {"name": "Qwen3 default",    "glob": "*qwen3*",  "params": {"reasoning_effort": "medium"}},
        {"name": "Gemma4 finetunes", "glob": "*gemma-4*","params": {}},
    ]
    PRESETS_FILE.write_text(json.dumps(defaults, indent=2))
    return defaults


def save_presets(presets: list):
    PRESETS_FILE.write_text(json.dumps(presets, indent=2))


# ── VRAM estimate ───────────────────────────────────────────────
class EstimateRequest(BaseModel):
    model_path: str
    n_gpu_layers: int
    ctx: int
    ctx_quant_q8: bool = False
    swa_full: bool = False
    mmproj_path: str = ""
    mtp_path: str = ""
    dflash_path: str = ""


def _file_size(path: str) -> int:
    if path:
        try:
            return os.path.getsize(path)
        except OSError:
            pass
    return 0


def _estimate_kv_cache(meta, n_ctx: int, elem_bytes: float, swa_full: bool,
                       n_ubatch: int = 512, n_seq_max: int = 1) -> int:
    if not meta.has_swa:
        cells = n_ctx
        return int(
            2 * cells * meta.n_layer
            * meta.n_head_kv
            * (meta.head_dim_k + meta.head_dim_v)
            * elem_bytes
        )

    swa_layers = meta.swa_layers()
    n_swa = len(swa_layers)
    n_full = meta.n_layer - n_swa

    if swa_full:
        cells_swa = n_ctx
    else:
        cells_swa = min(n_ctx, meta.swa_window * n_seq_max + n_ubatch)
        cells_swa = ((cells_swa + 255) // 256) * 256

    kv_per_cell_full = meta.n_head_kv * (meta.head_dim_k + meta.head_dim_v)
    kv_per_cell_swa  = meta.n_head_kv_swa * (meta.head_dim_k_swa + meta.head_dim_v_swa)

    kv_full = n_full * n_ctx     * kv_per_cell_full * elem_bytes
    kv_swa  = n_swa  * cells_swa * kv_per_cell_swa  * elem_bytes

    return int(kv_full + kv_swa)


# ── App ─────────────────────────────────────────────────────────
_model_cache: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_cache
    try:
        _model_cache = scan_models(_model_dirs)
    except Exception as e:
        print(f"[launcher] error in initial scan: {e}")
        _model_cache = []
    print(f"[launcher] Indexed {len(_model_cache)} models from {_model_dirs}")
    yield


app = FastAPI(title="Llama.cpp Launcher", lifespan=lifespan)


@app.get("/api/models")
def api_models():
    return [asdict(m) for m in _model_cache]


@app.post("/api/rescan")
def api_rescan():
    global _model_cache
    try:
        _model_cache = scan_models(_model_dirs)
    except Exception as e:
        print(f"[launcher] error in rescan: {e}")
        return {"error": str(e), "count": len(_model_cache)}
    return {"count": len(_model_cache)}


class ConfigIn(BaseModel):
    model_dirs: Optional[list] = None
    backends: Optional[dict] = None
    launch_options: Optional[dict] = None


@app.get("/api/config")
def api_config_get():
    return _config


@app.put("/api/config")
def api_config_put(p: ConfigIn):
    global _config, _model_dirs, _model_cache
    if p.model_dirs is not None:
        _config["model_dirs"] = p.model_dirs
        _model_dirs = list(p.model_dirs)
    if p.backends is not None:
        _config["backends"] = p.backends
    if p.launch_options is not None:
        _config["launch_options"] = p.launch_options
    save_config(_config)
    try:
        _model_cache = scan_models(_model_dirs)
    except Exception as e:
        print(f"[launcher] error in rescan after config: {e}")
        return {"ok": True, "count": len(_model_cache), "error": str(e)}
    return {"ok": True, "count": len(_model_cache)}


class ModelPathUpdate(BaseModel):
    mtp_path: Optional[str] = None
    mmproj_path: Optional[str] = None


@app.put("/api/models/{name}/paths")
def api_model_paths_put(name: str, p: ModelPathUpdate):
    global _overrides
    target = next((m for m in _model_cache if m.name == name), None)
    if target is None:
        return {"error": f"Model '{name}' not found"}

    ov = _overrides.setdefault(name, {})
    if p.mtp_path is not None:
        ov["mtp_path"] = p.mtp_path
    if p.mmproj_path is not None:
        ov["mmproj_path"] = p.mmproj_path
    save_overrides(_overrides)

    # Apply immediately to the in-memory cache
    if p.mtp_path is not None:
        target.mtp_path = p.mtp_path
    if p.mmproj_path is not None:
        target.mmproj_path = p.mmproj_path

    return {"ok": True}


@app.get("/api/presets")
def api_presets_get():
    return load_presets()


@app.post("/api/presets")
def api_presets_post(p: PresetIn):
    presets = load_presets()
    presets = [x for x in presets if x["name"] != p.name]
    presets.append(p.model_dump())
    save_presets(presets)
    return p.model_dump()


@app.delete("/api/presets/{name}")
def api_presets_delete(name: str):
    presets = [x for x in load_presets() if x["name"] != name]
    save_presets(presets)
    return {"ok": True}


@app.post("/api/estimate")
def api_estimate(req: EstimateRequest):
    meta = read_gguf_meta(req.model_path)
    if meta is None:
        return {"error": "Could not read the GGUF header of this file"}

    total_size = meta.total_file_size
    for m in _model_cache:
        if m.path == req.model_path:
            total_size = m.file_size
            break

    n_layer = meta.n_layer
    layers_on_gpu = max(0, min(req.n_gpu_layers, n_layer))

    if layers_on_gpu >= n_layer:
        weights_gpu_bytes = total_size
    else:
        parsed_total = meta.block_bytes + meta.non_block_bytes
        block_frac = (meta.block_bytes / parsed_total) if parsed_total > 0 else 0.95
        weights_gpu_bytes = int(total_size * block_frac * (layers_on_gpu / n_layer))

    elem_bytes = 1.0625 if req.ctx_quant_q8 else 2.0
    kv_bytes = _estimate_kv_cache(
        meta,
        n_ctx=max(0, req.ctx),
        elem_bytes=elem_bytes,
        swa_full=req.swa_full,
    )

    mmproj_bytes = _file_size(req.mmproj_path)
    mtp_bytes    = _file_size(req.mtp_path)
    dflash_bytes = _file_size(req.dflash_path)

    total_bytes = weights_gpu_bytes + kv_bytes + mmproj_bytes + mtp_bytes + dflash_bytes
    G = 1024 ** 3

    return {
        "arch": meta.arch,
        "n_layer": meta.n_layer,
        "n_embd": meta.n_embd,
        "n_head": meta.n_head,
        "n_head_kv": meta.n_head_kv,
        "layers_on_gpu": layers_on_gpu,
        "is_moe": meta.is_moe,
        "n_expert": meta.n_expert,
        "n_expert_used": meta.n_expert_used,
        "file_type_name": meta.file_type_name,
        "context_length": meta.context_length,
        "has_embedded_mtp": meta.has_embedded_mtp,
        "has_swa": meta.has_swa,
        "swa_window": meta.swa_window,
        "n_swa_layers": len(meta.swa_layers()) if meta.has_swa else 0,
        "n_full_layers": (meta.n_layer - len(meta.swa_layers())) if meta.has_swa else meta.n_layer,
        "weights_gib": round(weights_gpu_bytes / G, 2),
        "kv_gib":      round(kv_bytes / G, 2),
        "mmproj_gib":  round(mmproj_bytes / G, 2),
        "mtp_gib":     round(mtp_bytes / G, 2),
        "dflash_gib":  round(dflash_bytes / G, 2),
        "total_gib":   round(total_bytes / G, 2),
    }


@app.get("/")
def index():
    return FileResponse(HTML_FILE)


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
