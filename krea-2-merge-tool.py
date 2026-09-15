#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safetensors Model Merge Tool (Krea 2 / ComfyUI)
-----------------------------------------------
Merge two .safetensors models with RAM control (tensor by tensor),
support for fp8 (F8_E4M3) and int8 quantized tensors (ComfyUI format:
weight + weight_scale + comfy_quant), optional GPU (CUDA) with CPU fallback.

Output formats:
  - auto: preserves the original dtype of each tensor (requantizes int8)
  - int8: quantizes conv/linear weights to int8 with per-row scale
  - fp8 : converts to float8_e4m3fn
  - fp16: converts to float16

Usage:  python merge_tool.py
"""

import gc
import json
import os
import queue
import struct
import threading
import traceback
import argparse

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

import torch

try:
    from safetensors import safe_open
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DTYPES_ST = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
DTYPES_ST_REV = {v: k for k, v in DTYPES_ST.items()}
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}
# dtype used in merge calculations
MATH_DTYPE = torch.float32
# byte limit for processing a tensor on GPU (avoids VRAM OOM)
GPU_MAX_TENSOR_BYTES = 512 * 1024 * 1024


# ----------------------------------------------------------------------------
# Safetensors header reading (without loading weights)
# ----------------------------------------------------------------------------
def read_header(path):
    """Reads the JSON header of a .safetensors file. Returns (header, header_len)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


def tensor_infos(header):
    """Returns {name: info} without the __metadata__."""
    return {k: v for k, v in header.items() if k != "__metadata__"}


# ----------------------------------------------------------------------------
# Individual tensor reading (streaming, without loading the whole file)
# ----------------------------------------------------------------------------
class TensorReader:
    """Reads tensors individually from a .safetensors via mmap."""

    def __init__(self, path):
        self.path = path
        header, self.header_len = read_header(path)
        self.header = header
        self.infos = tensor_infos(header)
        self._file = None
        self._mmap = None

    def _ensure_open(self):
        if self._file is None:
            import mmap
            self._file = open(self.path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def read(self, name):
        """Reads a tensor as torch.Tensor (original dtype)."""
        self._ensure_open()
        info = self.infos[name]
        dt = DTYPES_ST[info["dtype"]]
        start, end = info["data_offsets"]
        nbytes = end - start
        buf = self._mmap[self.header_len + start: self.header_len + end]
        t = torch.frombuffer(bytearray(buf), dtype=dt).reshape(info["shape"])
        return t

    def read_raw(self, name):
        """Reads raw bytes of a tensor (to copy without converting)."""
        self._ensure_open()
        info = self.infos[name]
        start, end = info["data_offsets"]
        return self._mmap[self.header_len + start: self.header_len + end]

    def close(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None


# ----------------------------------------------------------------------------
# Dequantization / Quantization (ComfyUI int8 format)
# ----------------------------------------------------------------------------
def dequantize_int8(weight_i8, scale, math_dtype=MATH_DTYPE):
    """weight (I8, [out, in]) * scale (F32, [out, 1]) -> math_dtype."""
    return weight_i8.to(math_dtype) * scale.to(math_dtype)


def quantize_int8(weight_f32, math_dtype=MATH_DTYPE):
    """Quantizes to int8 with per-row scale. Returns (i8, scale_f32)."""
    w = weight_f32.to(math_dtype)
    if w.dim() == 1:
        w = w.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    if squeeze:
        q = q.squeeze(1)
        scale = scale.squeeze(1).unsqueeze(1)  # scale [n,1] like the originals
    else:
        scale = scale.reshape(-1, 1)
    return q, scale.to(torch.float32)


def cast_float(t, dtype_key):
    """Converts a float tensor to the target dtype."""
    if dtype_key == "F8_E4M3":
        return t.to(torch.float8_e4m3fn)
    if dtype_key == "F8_E5M2":
        return t.to(torch.float8_e5m2)
    return t.to(DTYPES_ST[dtype_key])


def tensor_to_bytes(t):
    """Serializes a tensor to bytes in safetensors format (little-endian)."""
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.view(torch.uint8).numpy().tobytes()
    return t.numpy().tobytes()


def _comfy_quant_bytes():
    """Generates the comfy_quant descriptor (JSON bytes) for int8_tensorwise."""
    desc = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    return json.dumps(desc, separators=(",", ":")).encode("utf-8")


# ----------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------
def merge_values(a, b, wa, wb, method):
    """Merges two float tensors. method: 'linear' or 'weighted_sum'."""
    if method == "weighted_sum":
        total = wa + wb
        if total == 0:
            return a
        wa, wb = wa / total, wb / total
    return a * wa + b * wb


class MergeJob:
    """Runs the merge tensor by tensor and writes the output in streaming."""

    def __init__(self, path_a, path_b, out_path, weight_a, weight_b,
                 method="linear", out_format="auto", use_gpu=True,
                 math_dtype=MATH_DTYPE, gpu_max_tensor_bytes=GPU_MAX_TENSOR_BYTES,
                 keep_metadata=True, custom_meta_tag=None,
                 log_fn=None, progress_fn=None, cancel_flag=None):
        self.path_a = path_a
        self.path_b = path_b
        self.out_path = out_path
        self.wa = weight_a / 100.0
        self.wb = weight_b / 100.0
        self.method = method
        self.out_format = out_format
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.math_dtype = math_dtype
        self.gpu_max_tensor_bytes = gpu_max_tensor_bytes
        self.keep_metadata = keep_metadata
        self.custom_meta_tag = custom_meta_tag
        self.log = log_fn or (lambda msg: None)
        self.progress = progress_fn or (lambda cur, total, name: None)
        self.cancel_flag = cancel_flag or (lambda: False)

    # ------------------------------------------------------------------ utils
    def _to_device(self, t):
        nbytes = t.numel() * t.element_size()
        if self.use_gpu and nbytes <= self.gpu_max_tensor_bytes:
            return t.to(self.device, non_blocking=True)
        return t

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ keys
    def _normalize_key(self, key):
        """Strips the common CI prefix so keys from different exporters match."""
        for p in ("model.diffusion_model.", "diffusion_model."):
            if key.startswith(p):
                return key[len(p):]
        return key

    def _key_maps(self, ra, rb):
        """Map normalized key -> actual key for each reader."""
        la = {self._normalize_key(k): k for k in ra.infos}
        lb = {self._normalize_key(k): k for k in rb.infos}
        return la, lb

    # ------------------------------------------------------------- validation
    def validate(self):
        ra = TensorReader(self.path_a)
        rb = TensorReader(self.path_b)
        try:
            la, lb = self._key_maps(ra, rb)
            self._la, self._lb = la, lb
            self._ra, self._rb = ra, rb

            # "core" names: main weights only (exclude the _scale companion and
            # the comfy_quant descriptor, both regenerated during the merge).
            def core(n):
                return not n.endswith("_scale") and not n.endswith(".comfy_quant")

            ka = {n for n in la if core(n)}
            kb = {n for n in lb if core(n)}
            only_a = ka - kb
            only_b = kb - ka
            if only_a:
                self.log(f"Warning: {len(only_a)} tensors only exist in A (will be kept from A).")
            if only_b:
                self.log(f"Warning: {len(only_b)} tensors only exist in B (will be kept from B).")

            union = ka | kb
            overlap_pct = (len(ka & kb) / len(union)) if union else 1.0
            if overlap_pct < 0.90:
                details = []
                for n in sorted(only_a | only_b)[:8]:
                    if n in only_a:
                        details.append(f"  • {n}  (only in A, shape={ra.infos[la[n]]['shape']})")
                    else:
                        details.append(f"  • {n}  (only in B, shape={rb.infos[lb[n]]['shape']})")
                if len(only_a | only_b) > 8:
                    details.append(f"  … and {len(only_a | only_b) - 8} more.")
                raise ValueError(
                    f"Tensor names overlap between A and B is only {overlap_pct:.0%} "
                    f"({len(ka & kb)} of {len(union)}). "
                    f"{len(only_a)} tensors exist only in A and {len(only_b)} only in B.\n"
                    + "This usually means different architectures/converters. Merging "
                      "would NOT merge the weights — it would pack both sets into the "
                      "same file (doubling its size) instead of combining them.\n"
                    + chr(10).join(details)
                    + "\nConfirm that A and B are the same model/architecture before continuing."
                )

            mismatch = []
            for n in ka & kb:
                ia = ra.infos[la[n]]
                ib = rb.infos[lb[n]]
                if ia["shape"] != ib["shape"]:
                    mismatch.append((n, ia, ib))
            if mismatch:
                details = []
                for k, ia, ib in mismatch[:8]:
                    details.append(
                        f"  • {k}\n      A: shape={ia['shape']} dtype={ia['dtype']}\n"
                        f"      B: shape={ib['shape']} dtype={ib['dtype']}")
                if len(mismatch) > 8:
                    details.append(f"  … and {len(mismatch) - 8} more.")
                raise ValueError(
                    f"{len(mismatch)} tensors have different shapes — the models are not "
                    f"compatible for merging.\n{chr(10).join(details)}")
            return ra, rb
        except Exception:
            ra.close()
            rb.close()
            raise

    # ------------------------------------------------------------------ merge
    def run(self):
        for label, path in (("A", self.path_a), ("B", self.path_b)):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Model {label} not found: {path}")
            if os.path.getsize(path) == 0:
                raise ValueError(
                    f"Model {label} ({path}) is empty (0 bytes).")
        try:
            ra, rb = self.validate()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not open model: {getattr(e, 'filename', e)}") from e
        except (json.JSONDecodeError, struct.error) as e:
            raise ValueError(
                "Corrupted safetensors header in the input model.\n"
                f"Details: {e}") from e
        tmp_data_path = self.out_path + ".dat"
        try:
            return self._run_merge(ra, rb, tmp_data_path)
        finally:
            ra.close()
            rb.close()
            if os.path.exists(tmp_data_path):
                try:
                    os.remove(tmp_data_path)
                except OSError:
                    pass

    def _run_merge(self, ra, rb, tmp_data_path):
        # Iterate over normalized keys: A's tensors first, then B-only ones.
        names = list(self._la.keys()) + [k for k in self._lb if k not in self._la]
        total = len(names)

        out_header = {}
        offset = 0
        meta_a = ra.header.get("__metadata__", {})
        meta_b = rb.header.get("__metadata__", {})

        merged_meta = dict(meta_a) if self.keep_metadata else {}
        if self.keep_metadata:
            merged_meta.update(meta_b)
        merged_meta["merge_tool"] = "krea2_safetensors_merge_tool"
        merged_meta["merge_parents"] = json.dumps(
            [os.path.basename(self.path_a), os.path.basename(self.path_b)])
        merged_meta["merge_weights"] = json.dumps(
            {"a": self.wa, "b": self.wb, "method": self.method})
        merged_meta["merge_format"] = self.out_format
        if self.custom_meta_tag:
            merged_meta["merge_tag"] = self.custom_meta_tag

        self.log(f"Device: {self.device} | Tensors: {total} | Method: {self.method} "
                 f"| Weights: A={self.wa:.0%} B={self.wb:.0%} | Format: {self.out_format}")

        with open(tmp_data_path, "wb") as data_f:
            for idx, name in enumerate(names):
                if self.cancel_flag():
                    raise InterruptedError("Cancelled by the user.")
                self.progress(idx, total, name)
                key_a = self._la.get(name)
                key_b = self._lb.get(name)
                try:
                    offset = self._process_one_tensor(
                        ra, rb, name, key_a, key_b, data_f, out_header, offset)
                except Exception as e:
                    raise type(e)(
                        f"[{name}] {e}{self._tensor_context(ra, rb, name, key_a, key_b)}"
                    ) from e

            self.progress(total, total, "writing final file...")

        # ---- assemble final file: [len][json header][data]
        final_header = dict(out_header)
        final_header["__metadata__"] = merged_meta
        header_bytes = json.dumps(final_header, separators=(",", ":")).encode("utf-8")
        # safetensors requires header padding to a multiple of 8 (optional, but safe)
        pad = (8 - (len(header_bytes) % 8)) % 8
        header_bytes += b" " * pad

        with open(tmp_data_path, "rb") as src, open(self.out_path, "wb") as dst:
            dst.write(struct.pack("<Q", len(header_bytes)))
            dst.write(header_bytes)
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        self.log(f"Done! File saved to: {self.out_path}")
        return self.out_path

    def _tensor_context(self, ra, rb, name, key_a=None, key_b=None):
        """Returns a compact context string (dtypes/shapes) for a failed tensor."""
        parts = []
        ia = ra.infos.get(key_a) if key_a else None
        ib = rb.infos.get(key_b) if key_b else None
        if ia:
            parts.append(f"A: shape={ia['shape']} dtype={ia['dtype']}")
        else:
            parts.append("A: (absent)")
        if ib:
            parts.append(f"B: shape={ib['shape']} dtype={ib['dtype']}")
        else:
            parts.append("B: (absent)")
        return f" | {' | '.join(parts)}"

    def _read_value(self, reader, key):
        """Reads a value tensor and dequantizes it to math dtype.
        Handles plain floats, fp8 (pure or fp8+scale) and int8+scale."""
        info = reader.infos[key]
        dtype = info["dtype"]
        scale_key = key + "_scale"
        has_scale = scale_key in reader.infos
        t = reader.read(key)
        if dtype == "I8" and has_scale:
            return dequantize_int8(t, reader.read(scale_key), self.math_dtype)
        if dtype in FLOAT_DTYPES:
            t = t.to(self.math_dtype)
            if has_scale:
                t = t * reader.read(scale_key).to(self.math_dtype)
            return t
        return t

    def _process_one_tensor(self, ra, rb, name, key_a, key_b,
                            data_f, out_header, offset):
        """Processes a single tensor (returns the new offset)."""
        info_a = ra.infos.get(key_a) if key_a else None
        info_b = rb.infos.get(key_b) if key_b else None
        in_a = key_a is not None
        in_b = key_b is not None

        # ---- scale tensor: written together with its weight; skip here.
        if name.endswith("_scale"):
            base = name[:-len("_scale")]
            if base in self._la or base in self._lb:
                return offset

        # ---- comfy_quant descriptor: only kept when the blob is int8 in the
        #      output (auto/int8). Dropped for fp8/fp16 output.
        if name.endswith(".comfy_quant"):
            base_norm = name[:-len(".comfy_quant")] + ".weight"
            if self.out_format in ("auto", "int8") and self._weight_is_i8(base_norm):
                raw = _comfy_quant_bytes()
                data_f.write(raw)
                out_header[name] = {
                    "dtype": "U8", "shape": [len(raw)],
                    "data_offsets": [offset, offset + len(raw)],
                }
                offset += len(raw)
            return offset

        offset = self._merge_tensor(
            ra, rb, name, key_a, key_b, info_a, info_b,
            data_f, out_header, offset)
        self._cleanup()
        return offset

    def _weight_is_i8(self, name):
        """True if the (normalized) weight `name` is int8 in the output."""
        if self.out_format not in ("auto", "int8"):
            return False
        key_a = self._la.get(name)
        key_b = self._lb.get(name)
        dtype_a = self._ra.infos[key_a]["dtype"] if key_a else None
        dtype_b = self._rb.infos[key_b]["dtype"] if key_b else None
        return dtype_a == "I8" or dtype_b == "I8"

    # ----------------------------------------------------------- tensor merge
    def _merge_tensor(self, ra, rb, name, key_a, key_b, info_a, info_b,
                      data_f, out_header, offset):
        """Dequant, merge and (re)quantize a weight. Returns the new offset."""
        wa = self._read_value(ra, key_a) if key_a else None
        wb = self._read_value(rb, key_b) if key_b else None

        # ---- non-float inputs (embeddings kept as-is): write through.
        if (wa is not None and not wa.is_floating_point()) or \
           (wb is not None and not wb.is_floating_point()):
            t = wa if wa is not None else wb
            raw = tensor_to_bytes(t)
            data_f.write(raw)
            out_header[name] = {
                "dtype": DTYPES_ST_REV.get(t.dtype, "F32"),
                "shape": list(t.shape),
                "data_offsets": [offset, offset + len(raw)]}
            del t
            return offset + len(raw)

        # ---- merge in float
        if wa is None:
            wa = wb.clone()
        if wb is None:
            wb = wa.clone()
        wa = self._to_device(wa)
        wb = self._to_device(wb)
        merged = merge_values(wa, wb, self.wa, self.wb, self.method).cpu()
        # release the dequantized inputs
        del wa, wb

        if self.out_format in ("auto", "int8"):
            q, scale = quantize_int8(merged, self.math_dtype)
            raw = tensor_to_bytes(q)
            data_f.write(raw)
            out_header[name] = {"dtype": "I8", "shape": list(q.shape),
                                "data_offsets": [offset, offset + len(raw)]}
            sraw = tensor_to_bytes(scale)
            data_f.write(sraw)
            out_header[name + "_scale"] = {
                "dtype": "F32", "shape": list(scale.shape),
                "data_offsets": [offset + len(raw), offset + len(raw) + len(sraw)]}
            new_offset = offset + len(raw) + len(sraw)
        else:
            key = "F16" if self.out_format == "fp16" else "F8_E4M3"
            t = cast_float(merged, key)
            raw = tensor_to_bytes(t)
            data_f.write(raw)
            out_header[name] = {"dtype": key, "shape": list(t.shape),
                                "data_offsets": [offset, offset + len(raw)]}
            new_offset = offset + len(raw)
        del merged
        return new_offset


# ----------------------------------------------------------------------------
# Tkinter GUI
# ----------------------------------------------------------------------------
if _HAS_TK:
    class MergeApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Safetensors Model Merge — Krea 2")
            self.minsize(720, 600)
            # Auto-size to fit all content so the bottom widgets are never clipped.
            self.update_idletasks()
            self.geometry("")

            self.msg_queue = queue.Queue()
            self.cancel_event = threading.Event()
            self.worker = None

            self._build_ui()
            self.after(100, self._poll_queue)

        # ------------------------------------------------------------- UI
        def _build_ui(self):
            pad = {"padx": 8, "pady": 3}
            main = ttk.Frame(self)
            main.pack(fill="both", expand=True)
            main.columnconfigure(0, weight=1)
            main.columnconfigure(1, weight=1)
            main.rowconfigure(0, weight=1)

            # --- Left column: controls
            left = ttk.Frame(main)
            left.grid(row=0, column=0, sticky="nsew")
            left.columnconfigure(0, weight=1)

            # --- Models (A and B in a single frame)
            frm_m = ttk.LabelFrame(left, text="Models")
            frm_m.grid(row=0, column=0, sticky="ew", **pad)
            frm_m.columnconfigure(1, weight=1)

            ttk.Label(frm_m, text="Model A:").grid(row=0, column=0, padx=6, pady=3)
            self.var_a = tk.StringVar()
            self.cmb_a = ttk.Combobox(frm_m, textvariable=self.var_a, state="readonly")
            self.cmb_a.grid(row=0, column=1, sticky="ew", padx=4)
            ttk.Button(frm_m, text="Browse...", command=lambda: self._browse(self.var_a)
                       ).grid(row=0, column=2, padx=4)

            ttk.Label(frm_m, text="Model B:").grid(row=1, column=0, padx=6, pady=3)
            self.var_b = tk.StringVar()
            self.cmb_b = ttk.Combobox(frm_m, textvariable=self.var_b, state="readonly")
            self.cmb_b.grid(row=1, column=1, sticky="ew", padx=4)
            ttk.Button(frm_m, text="Browse...", command=lambda: self._browse(self.var_b)
                       ).grid(row=1, column=2, padx=4)

            # --- Merge ratio + Output file (single frame)
            frm_wo = ttk.LabelFrame(left, text="Merge & Output")
            frm_wo.grid(row=1, column=0, sticky="ew", **pad)
            frm_wo.columnconfigure(1, weight=1)

            self.var_wa = tk.DoubleVar(value=50.0)
            self.var_wb = tk.DoubleVar(value=50.0)
            self._sync_lock = False

            ttk.Label(frm_wo, text="Model A:").grid(row=0, column=0, padx=6, pady=3)
            self.scl_a = ttk.Scale(frm_wo, from_=0, to=100, variable=self.var_wa,
                                   command=lambda v: self._sync_weights("a"))
            self.scl_a.grid(row=0, column=1, sticky="ew", padx=4)
            self.lbl_a = ttk.Label(frm_wo, text="50%", width=6)
            self.lbl_a.grid(row=0, column=2, padx=6)

            ttk.Label(frm_wo, text="Model B:").grid(row=1, column=0, padx=6, pady=3)
            self.scl_b = ttk.Scale(frm_wo, from_=0, to=100, variable=self.var_wb,
                                   command=lambda v: self._sync_weights("b"))
            self.scl_b.grid(row=1, column=1, sticky="ew", padx=4)
            self.lbl_b = ttk.Label(frm_wo, text="50%", width=6)
            self.lbl_b.grid(row=1, column=2, padx=6)

            ttk.Label(frm_wo, text="Output:").grid(row=2, column=0, padx=6, pady=3)
            self.var_out = tk.StringVar()
            ttk.Entry(frm_wo, textvariable=self.var_out).grid(
                row=2, column=1, sticky="ew", padx=4)
            ttk.Button(frm_wo, text="Save as...",
                       command=self._browse_out).grid(row=2, column=2, padx=4)

            # --- Options (Notebook: Basic / Advanced)
            frm_o = ttk.LabelFrame(left, text="Options")
            frm_o.grid(row=2, column=0, sticky="ew", **pad)
            nb = ttk.Notebook(frm_o)
            nb.pack(fill="x", padx=6, pady=3)

            # ---- Basic tab
            tab_basic = ttk.Frame(nb)
            nb.add(tab_basic, text="Basic")
            ttk.Label(tab_basic, text="Output format:").grid(
                row=0, column=0, sticky="w", padx=6, pady=2)
            self.var_fmt = tk.StringVar(value="auto")
            ttk.Combobox(tab_basic, textvariable=self.var_fmt, state="readonly", width=22,
                         values=["auto", "int8", "fp8", "fp16"]).grid(
                row=0, column=1, sticky="w", padx=6, pady=2)

            # ---- Advanced tab
            tab_adv = ttk.Frame(nb)
            nb.add(tab_adv, text="Advanced")

            ttk.Label(tab_adv, text="Method:").grid(
                row=0, column=0, sticky="w", padx=6, pady=2)
            self.var_method = tk.StringVar(value="linear")
            ttk.Combobox(tab_adv, textvariable=self.var_method, state="readonly", width=22,
                         values=["linear", "weighted_sum"]).grid(
                row=0, column=1, sticky="w", padx=6, pady=2)

            ttk.Label(tab_adv, text="Merge math dtype:").grid(
                row=1, column=0, sticky="w", padx=6, pady=2)
            self.var_math = tk.StringVar(value="F32")
            ttk.Combobox(tab_adv, textvariable=self.var_math, state="readonly", width=22,
                         values=["F32", "F16", "BF16"]).grid(
                row=1, column=1, sticky="w", padx=6, pady=2)

            ttk.Label(tab_adv, text="GPU max tensor (MB):").grid(
                row=2, column=0, sticky="w", padx=6, pady=2)
            self.var_gpu_mb = tk.StringVar(
                value=str(GPU_MAX_TENSOR_BYTES // (1024 * 1024)))
            ttk.Entry(tab_adv, textvariable=self.var_gpu_mb, width=22).grid(
                row=2, column=1, sticky="w", padx=6, pady=2)

            self.var_gpu = tk.BooleanVar(value=torch.cuda.is_available())
            self.chk_gpu = ttk.Checkbutton(
                tab_adv, text="Use GPU (CUDA)" + ("" if torch.cuda.is_available() else " — not available"),
                variable=self.var_gpu,
                state="normal" if torch.cuda.is_available() else "disabled")
            self.chk_gpu.grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=2)

            self.var_keep_meta = tk.BooleanVar(value=True)
            ttk.Checkbutton(tab_adv, text="Keep parent metadata",
                            variable=self.var_keep_meta).grid(
                row=4, column=0, columnspan=2, sticky="w", padx=6, pady=2)

            ttk.Label(tab_adv, text="Custom metadata tag:").grid(
                row=5, column=0, sticky="w", padx=6, pady=2)
            self.var_meta_tag = tk.StringVar()
            ttk.Entry(tab_adv, textvariable=self.var_meta_tag, width=22).grid(
                row=5, column=1, sticky="w", padx=6, pady=2)

            tab_adv.columnconfigure(1, weight=1)

            # --- Progress + Buttons (single frame)
            frm_pb = ttk.LabelFrame(left, text="Progress")
            frm_pb.grid(row=3, column=0, sticky="ew", **pad)
            self.progress = ttk.Progressbar(frm_pb, mode="determinate")
            self.progress.pack(fill="x", padx=6, pady=3)
            self.lbl_prog = ttk.Label(frm_pb, text="Ready.")
            self.lbl_prog.pack(anchor="w", padx=6, pady=(0, 3))
            self.btn_start = ttk.Button(frm_pb, text="Start Merge", command=self._start)
            self.btn_start.pack(side="left", padx=4)
            self.btn_cancel = ttk.Button(frm_pb, text="Cancel",
                                         command=self._cancel, state="disabled")
            self.btn_cancel.pack(side="left", padx=4)

            # --- Log (right column, expands to fill the vertical space)
            frm_log = ttk.LabelFrame(main, text="Log")
            frm_log.grid(row=0, column=1, sticky="nsew", **pad)
            self.txt_log = tk.Text(frm_log, height=6, state="disabled", wrap="none")
            sb = ttk.Scrollbar(frm_log, command=self.txt_log.yview)
            self.txt_log.configure(yscrollcommand=sb.set)
            self.txt_log.pack(fill="both", expand=True, side="left", padx=6, pady=6)
            sb.pack(fill="y", side="right", pady=6)

            self._populate_models()

        def _populate_models(self):
            """Lists the .safetensors from the script directory and the cwd."""
            dirs = {
                os.path.normcase(os.path.realpath(
                    os.path.dirname(os.path.abspath(__file__)))),
                os.path.normcase(os.path.realpath(os.getcwd())),
            }
            files = []
            for d in dirs:
                if os.path.isdir(d):
                    files += [os.path.join(d, f) for f in os.listdir(d)
                              if f.lower().endswith(".safetensors")]
            files = sorted(set(files))
            self.cmb_a["values"] = files
            self.cmb_b["values"] = files
            if files:
                self.cmb_a.current(0)
                self.cmb_b.current(min(1, len(files) - 1))

        # ------------------------------------------------------------ helpers
        def _browse(self, var):
            p = filedialog.askopenfilename(
                title="Select model",
                filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")])
            if p:
                var.set(p)

        def _browse_out(self):
            p = filedialog.asksaveasfilename(
                title="Save merged model", defaultextension=".safetensors",
                filetypes=[("Safetensors", "*.safetensors")])
            if p:
                self.var_out.set(p)

        def _sync_weights(self, changed):
            if self._sync_lock:
                return
            self._sync_lock = True
            try:
                if changed == "a":
                    va = self.var_wa.get()
                    self.var_wb.set(round(100.0 - va, 1))
                else:
                    vb = self.var_wb.get()
                    self.var_wa.set(round(100.0 - vb, 1))
                self.lbl_a.config(text=f"{self.var_wa.get():.0f}%")
                self.lbl_b.config(text=f"{self.var_wb.get():.0f}%")
            finally:
                self._sync_lock = False

        def _log(self, msg):
            self.msg_queue.put(("log", msg))

        def _poll_queue(self):
            try:
                while True:
                    kind, payload = self.msg_queue.get_nowait()
                    if kind == "log":
                        self.txt_log.configure(state="normal")
                        self.txt_log.insert("end", payload + "\n")
                        self.txt_log.see("end")
                        self.txt_log.configure(state="disabled")
                    elif kind == "progress":
                        cur, total, name = payload
                        self.progress.configure(maximum=total, value=cur)
                        self.lbl_prog.config(text=f"[{cur}/{total}] {name}")
                    elif kind == "done":
                        self.progress.configure(value=self.progress["maximum"])
                        self.lbl_prog.config(text="Done!")
                        self.btn_start.config(state="normal")
                        self.btn_cancel.config(state="disabled")
                        messagebox.showinfo("Merge", "Merge completed successfully!")
                    elif kind == "error":
                        self.lbl_prog.config(text="Error.")
                        self.btn_start.config(state="normal")
                        self.btn_cancel.config(state="disabled")
                        messagebox.showerror("Merge error", payload)
            except queue.Empty:
                pass
            self.after(100, self._poll_queue)

        # ------------------------------------------------------------ actions
        def _start(self):
            path_a = self.var_a.get()
            path_b = self.var_b.get()
            out_path = self.var_out.get()

            if not path_a or not os.path.isfile(path_a):
                messagebox.showerror("Error", "Select Model A.")
                return
            if not path_b or not os.path.isfile(path_b):
                messagebox.showerror("Error", "Select Model B.")
                return
            if path_a == path_b:
                messagebox.showerror("Error", "Models A and B must be different.")
                return
            if not out_path:
                messagebox.showerror("Error", "Set the output file.")
                return
            if os.path.exists(out_path) and not messagebox.askyesno(
                    "Overwrite", "The output file already exists. Overwrite?"):
                return

            self.cancel_event.clear()
            self.btn_start.config(state="disabled")
            self.btn_cancel.config(state="normal")
            self.progress.configure(value=0)
            self._log(f"Starting merge: A={os.path.basename(path_a)} "
                      f"({self.var_wa.get():.0f}%) + B={os.path.basename(path_b)} "
                      f"({self.var_wb.get():.0f}%)")

            job = MergeJob(
                path_a, path_b, out_path,
                weight_a=self.var_wa.get(), weight_b=self.var_wb.get(),
                method=self.var_method.get(), out_format=self.var_fmt.get(),
                use_gpu=self.var_gpu.get(),
                math_dtype=DTYPES_ST.get(self.var_math.get(), MATH_DTYPE),
                gpu_max_tensor_bytes=self._gpu_max_bytes(),
                keep_metadata=self.var_keep_meta.get(),
                custom_meta_tag=self.var_meta_tag.get().strip() or None,
                log_fn=self._log,
                progress_fn=lambda c, t, n: self.msg_queue.put(("progress", (c, t, n))),
                cancel_flag=self.cancel_event.is_set,
            )
            self.worker = threading.Thread(target=self._run_job, args=(job,), daemon=True)
            self.worker.start()

        def _run_job(self, job):
            try:
                job.run()
                self.msg_queue.put(("done", None))
            except InterruptedError:
                self.msg_queue.put(("log", "Merge cancelled."))
                self.msg_queue.put(("error", "Merge cancelled by the user."))
            except Exception as e:
                tb = traceback.format_exc()
                self.msg_queue.put(("log", tb))
                self.msg_queue.put(("error", f"{type(e).__name__}: {e}"))

        def _cancel(self):
            self.cancel_event.set()
            self._log("Cancelling... (may take a few seconds)")

        def _gpu_max_bytes(self):
            """Parses the GPU max tensor size (MB) field, falling back to the default."""
            try:
                mb = float(self.var_gpu_mb.get())
                if mb > 0:
                    return int(mb * 1024 * 1024)
            except (ValueError, tk.TclError):
                pass
            return GPU_MAX_TENSOR_BYTES


def main():
    parser = argparse.ArgumentParser(description="Merge safetensors models")
    parser.add_argument("--gui", action="store_true", help="Force graphical interface")
    parser.add_argument("--a", help="Model A (.safetensors)")
    parser.add_argument("--b", help="Model B (.safetensors)")
    parser.add_argument("--out", help="Output file")
    parser.add_argument("--wa", type=float, default=50.0, help="Weight of model A (%%)")
    parser.add_argument("--wb", type=float, default=50.0, help="Weight of model B (%%)")
    parser.add_argument("--method", default="linear", choices=["linear", "weighted_sum"])
    parser.add_argument("--format", default="auto", choices=["auto", "int8", "fp8", "fp16"])
    parser.add_argument("--math-dtype", default="F32", choices=["F32", "F16", "BF16"],
                        help="Dtype used in merge calculations")
    parser.add_argument("--gpu-max-bytes", type=int, default=GPU_MAX_TENSOR_BYTES,
                        help="Max tensor size (bytes) processed on GPU")
    parser.add_argument("--keep-metadata", action="store_true", default=True,
                        help="Preserve __metadata__ from the source models")
    parser.add_argument("--no-keep-metadata", dest="keep_metadata", action="store_false",
                        help="Do not preserve source __metadata__")
    parser.add_argument("--meta-tag", default=None, help="Custom tag added to output metadata")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    if args.gui or (not args.a and not args.b):
        if not _HAS_TK:
            print("tkinter is not available in this Python. Use CLI mode:")
            print("  python merge_tool.py --a A.safetensors --b B.safetensors "
                  "--out out.safetensors --wa 60 --wb 40")
            return 1
        app = MergeApp()
        app.mainloop()
        return 0

    if not (args.a and args.b and args.out):
        parser.error("--a, --b and --out are required in CLI mode.")
    job = MergeJob(
        args.a, args.b, args.out,
        weight_a=args.wa, weight_b=args.wb,
        method=args.method, out_format=args.format,
        use_gpu=not args.cpu,
        math_dtype=DTYPES_ST.get(args.math_dtype, MATH_DTYPE),
        gpu_max_tensor_bytes=args.gpu_max_bytes,
        keep_metadata=args.keep_metadata,
        custom_meta_tag=args.meta_tag,
        log_fn=print,
        progress_fn=lambda c, t, n: print(f"\r[{c}/{t}] {n[:60]}", end=""),
    )
    try:
        job.run()
        print()
    except Exception as e:
        print("\n[ERROR] Merge failed:")
        print(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
