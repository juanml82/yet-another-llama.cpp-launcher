#!/usr/bin/env python3
"""gguf_read.py — GGUF header parser for the VRAM estimator.

Supports hybrid attention (ISWA) by reading:
  - <arch>.attention.sliding_window
  - <arch>.attention.sliding_window_pattern (bool array or scalar period)
  - <arch>.attention.key_length_swa / value_length_swa
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


VT_U8, VT_I8, VT_U16, VT_I16, VT_U32, VT_I32, VT_F32, VT_BOOL = range(8)
VT_STR, VT_ARR, VT_U64, VT_I64, VT_F64 = range(8, 13)

_STRUCT_FMT = {
    VT_U8:  "B",  VT_I8:  "b",
    VT_U16: "H",  VT_I16: "h",
    VT_U32: "I",  VT_I32: "i",
    VT_F32: "f",
    VT_BOOL: "?",
    VT_U64: "Q",  VT_I64: "q",
    VT_F64: "d",
}

FILE_TYPE_MAP = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}

_BLOCK_QUANTS = {
    2:  (32, 18), 3:  (32, 20), 6:  (32, 22), 7:  (32, 24),
    8:  (32, 34), 9:  (32, 36), 10: (256, 84), 11: (256, 110),
    12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 34),
    20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    29: (256, 56),
}

_SIMPLE_TYPES = {
    0: 4, 1: 2, 24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 30: 2,
}


def _as_int(value, default=0) -> int:
    if isinstance(value, list):
        if not value:
            return default
        try:
            return max(int(v) for v in value if isinstance(v, (int, float)))
        except (ValueError, TypeError):
            return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _read_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    if length > (1 << 24):
        raise ValueError(f"string length {length} exceeds sanity limit")
    return f.read(length).decode("utf-8", errors="replace")


def _skip_string(f) -> None:
    length = struct.unpack("<Q", f.read(8))[0]
    f.seek(length, 1)


def _read_value(f, vtype: int):
    if vtype == VT_ARR:
        elem_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_len > (1 << 22):
            raise ValueError(f"array length {arr_len} exceeds sanity limit")

        if elem_type == VT_STR:
            if arr_len > 4096:
                for _ in range(arr_len):
                    _skip_string(f)
                return None
            return [_read_string(f) for _ in range(arr_len)]

        fmt = _STRUCT_FMT.get(elem_type)
        if fmt is None:
            raise ValueError(f"unknown array element type {elem_type}")
        size = struct.calcsize(fmt)
        if arr_len > 4096:
            f.seek(arr_len * size, 1)
            return None
        data = f.read(arr_len * size)
        return list(struct.unpack(f"<{arr_len}{fmt}", data))

    if vtype == VT_STR:
        return _read_string(f)

    fmt = _STRUCT_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"unknown value type {vtype}")
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]


def _tensor_bytes(ttype: int, n_elements: int) -> int:
    if ttype in _SIMPLE_TYPES:
        return n_elements * _SIMPLE_TYPES[ttype]
    if ttype in _BLOCK_QUANTS:
        bs, bb = _BLOCK_QUANTS[ttype]
        blocks = (n_elements + bs - 1) // bs
        return blocks * bb
    return n_elements * 2


@dataclass
class GgufMeta:
    arch: str = ""
    n_layer: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    head_dim_k: int = 0
    head_dim_v: int = 0
    # ISWA / hybrid attention
    swa_window: int = 0
    swa_pattern: Optional[list] = None
    head_dim_k_swa: int = 0
    head_dim_v_swa: int = 0
    n_head_kv_swa: int = 0
    # MoE
    n_expert: int = 0
    n_expert_used: int = 0
    context_length: int = 0
    file_type: int = -1
    file_type_name: str = ""
    total_file_size: int = 0
    tensor_count: int = 0
    block_bytes: int = 0
    non_block_bytes: int = 0
    nextn_predict_layers: int = 0

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 1

    @property
    def has_embedded_mtp(self) -> bool:
        return self.nextn_predict_layers >= 1

    @property
    def has_swa(self) -> bool:
        return self.swa_window > 0 and self.swa_pattern is not None

    def swa_layers(self) -> list:
        """Returns the list of 0-based indices of layers with sliding window."""
        if not self.has_swa:
            return []
        return [i for i, is_swa in enumerate(self.swa_pattern[:self.n_layer]) if is_swa]

    def full_attention_layers(self) -> list:
        if not self.has_swa:
            return list(range(self.n_layer))
        return [i for i, is_swa in enumerate(self.swa_pattern[:self.n_layer]) if not is_swa]


def _parse_swa_pattern(raw, n_layer: int) -> Optional[list]:
    """Converts the raw sliding_window_pattern value to a per-layer list of booleans.

    Accepted formats:
      - [True, True, True, True, True, False, ...]  (per-layer bool array)
      - [6] or 6                                     (scalar period)
    """
    if raw is None:
        return None

    # Scalar (int)
    if isinstance(raw, int):
        period = raw
        if period > 1:
            return [((i + 1) % period) != 0 for i in range(n_layer)]
        return None

    if not isinstance(raw, list) or not raw:
        return None

    # Direct bool array (Gemma 4, Qwen2, etc.)
    if all(isinstance(x, (bool, int)) and not isinstance(x, list) for x in raw):
        # Detect if it's a per-layer pattern (length ~n_layer)
        # or a scalar period bottled as a single-element list
        if len(raw) == 1:
            return _parse_swa_pattern(raw[0], n_layer)
        if len(raw) >= n_layer:
            return [bool(x) for x in raw[:n_layer]]

    return None


def read_gguf_meta(path: str | Path) -> Optional[GgufMeta]:
    p = Path(path)
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None

            version = struct.unpack("<I", f.read(4))[0]
            if version < 2 or version > 3:
                print(f"[gguf_read] {p.name}: unsupported GGUF version {version}")
                return None

            tensor_cnt = struct.unpack("<Q", f.read(8))[0]
            param_cnt  = struct.unpack("<Q", f.read(8))[0]

            if tensor_cnt > 1_000_000 or param_cnt > 100_000:
                print(f"[gguf_read] {p.name}: implausible counts")
                return None

            # 1. Metadata KV
            params: dict = {}
            for _ in range(param_cnt):
                key = _read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                val = _read_value(f, vtype)
                params[key] = val

            # 2. Tensor info
            block_bytes = 0
            non_block_bytes = 0
            for _ in range(tensor_cnt):
                tname = _read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                if n_dims > 8:
                    raise ValueError(f"n_dims={n_dims} exceeds sanity limit")
                shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                struct.unpack("<Q", f.read(8))

                numel = 1
                for s in shape:
                    numel *= s
                tsize = _tensor_bytes(ttype, numel)

                if tname.startswith("blk."):
                    block_bytes += tsize
                else:
                    non_block_bytes += tsize

            # 3. Extraction
            arch = str(params.get("general.architecture", "") or "")
            if not arch:
                return None
            pref = arch

            n_layer = _as_int(params.get(f"{pref}.block_count"), 0)
            if n_layer == 0:
                return None

            n_embd = _as_int(params.get(f"{pref}.embedding_length"), 0)
            n_head = _as_int(params.get(f"{pref}.attention.head_count"), 0)
            n_head_kv = _as_int(params.get(f"{pref}.attention.head_count_kv"), n_head)

            head_dim_k = _as_int(params.get(f"{pref}.attention.key_length"), 0)
            head_dim_v = _as_int(params.get(f"{pref}.attention.value_length"), 0)
            if head_dim_k == 0 and n_head > 0 and n_embd > 0:
                head_dim_k = n_embd // n_head
            if head_dim_v == 0:
                head_dim_v = head_dim_k

            # ── ISWA: sliding window ──
            swa_window = _as_int(params.get(f"{pref}.attention.sliding_window"), 0)
            swa_pattern = _parse_swa_pattern(
                params.get(f"{pref}.attention.sliding_window_pattern"),
                n_layer,
            )

            head_dim_k_swa = _as_int(params.get(f"{pref}.attention.key_length_swa"), 0)
            head_dim_v_swa = _as_int(params.get(f"{pref}.attention.value_length_swa"), 0)
            n_head_kv_swa = _as_int(params.get(f"{pref}.attention.head_count_kv_swa"), 0)

            if head_dim_k_swa == 0:
                head_dim_k_swa = head_dim_k
            if head_dim_v_swa == 0:
                head_dim_v_swa = head_dim_v
            if n_head_kv_swa == 0:
                n_head_kv_swa = n_head_kv

            n_expert      = _as_int(params.get(f"{pref}.expert_count"), 0)
            n_expert_used = _as_int(params.get(f"{pref}.expert_used_count"), 0)
            context_length = _as_int(params.get(f"{pref}.context_length"), 0)
            nextn = _as_int(params.get(f"{pref}.nextn_predict_layers"), 0)

            file_type = _as_int(params.get("general.file_type"), -1)
            file_type_name = FILE_TYPE_MAP.get(file_type, "")

            return GgufMeta(
                arch=arch,
                n_layer=n_layer,
                n_embd=n_embd,
                n_head=n_head,
                n_head_kv=n_head_kv,
                head_dim_k=head_dim_k,
                head_dim_v=head_dim_v,
                swa_window=swa_window,
                swa_pattern=swa_pattern,
                head_dim_k_swa=head_dim_k_swa,
                head_dim_v_swa=head_dim_v_swa,
                n_head_kv_swa=n_head_kv_swa,
                n_expert=n_expert,
                n_expert_used=n_expert_used,
                context_length=context_length,
                file_type=file_type,
                file_type_name=file_type_name,
                total_file_size=p.stat().st_size,
                tensor_count=tensor_cnt,
                block_bytes=block_bytes,
                non_block_bytes=non_block_bytes,
                nextn_predict_layers=nextn,
            )
    except Exception as e:
        print(f"[gguf_read] {p.name}: {type(e).__name__}: {e}")
        return None
