#!/usr/bin/env python3
"""Merge per-FSDB TOG cache manifests into one global reduction cache.

This is intentionally dependency-light: it rewrites CSC sparse NPZ files using
only Python's standard library. The output manifest can be consumed by
code-ai/reduction/dimensionality_reduction_test.py --input-cache-manifest.
"""

from __future__ import annotations

import argparse
import ast
import gc
import sys
import json
import os
import re
import shutil
import struct
import zipfile
from array import array
from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path


def safe_stem(path: str) -> str:
    return Path(path).stem.replace(os.sep, "_")


def mkdirp(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def load_layout(manifest_path: Path) -> list[dict]:
    layout_path = manifest_path.parent / "expanded_layout.jsonl"
    if not layout_path.is_file():
        raise FileNotFoundError(f"missing expanded_layout.jsonl next to {manifest_path}")
    records: list[dict] = []
    with layout_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ordered_names_for_block(block: dict, layout: list[dict]) -> list[str]:
    raw_set = set(int(x) for x in block["raw_indices"])
    return [r["name"] for r in layout if int(r["raw_idx"]) in raw_set]


def parse_mem_to_bytes(value: str) -> int:
    s = str(value).strip().lower()
    m = re.fullmatch(r"([0-9.]+)\s*([kmgt]?b?)?", s)
    if not m:
        raise ValueError(f"bad memory size: {value}")
    val = float(m.group(1))
    unit = m.group(2) or "b"
    if unit in ("t", "tb"):
        return int(val * 1024**4)
    if unit in ("g", "gb"):
        return int(val * 1024**3)
    if unit in ("m", "mb"):
        return int(val * 1024**2)
    if unit in ("k", "kb"):
        return int(val * 1024)
    return int(val)


def check_memory_budget(label: str, estimated_bytes: int, memory_limit_bytes: int) -> None:
    if memory_limit_bytes <= 0:
        return
    if estimated_bytes > memory_limit_bytes:
        raise MemoryError(
            f"{label}: estimated peak {estimated_bytes / 1024**3:.2f}GiB exceeds "
            f"merge memory limit {memory_limit_bytes / 1024**3:.2f}GiB; "
            f"increase --max-mem/--merge-mem-frac or split inputs into smaller cache blocks"
        )


def parse_npy(blob: bytes) -> tuple[dict, bytes]:
    if blob[:6] != b"\x93NUMPY":
        raise ValueError("bad npy magic")
    major = blob[6]
    if major == 1:
        header_len = struct.unpack("<H", blob[8:10])[0]
        offset = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", blob[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"unsupported npy version: {major}")
    header = blob[offset:offset + header_len].decode("latin1")
    meta = ast.literal_eval(header)
    payload = blob[offset + header_len:]
    return meta, payload


def read_npy_meta_from_zip(zf: zipfile.ZipFile, name: str) -> tuple[dict, int]:
    with zf.open(name, "r") as f:
        prefix = f.read(10)
        if prefix[:6] != b"\x93NUMPY":
            raise ValueError(f"bad npy magic in {name}")
        major = prefix[6]
        if major == 1:
            header_len = struct.unpack("<H", prefix[8:10])[0]
        elif major in (2, 3):
            prefix += f.read(2)
            header_len = struct.unpack("<I", prefix[8:12])[0]
        else:
            raise ValueError(f"unsupported npy version in {name}: {major}")
        header = f.read(header_len).decode("latin1")
    meta = ast.literal_eval(header)
    return meta, zf.getinfo(name).file_size


def dtype_itemsize(descr: str) -> int:
    if descr.startswith("|S"):
        return int(descr[2:])
    if len(descr) >= 3 and descr[1] in "iu" and descr[2:].isdigit():
        return int(descr[2:])
    raise ValueError(f"unsupported dtype descr: {descr}")


def array_len_from_meta(meta: dict) -> int:
    shape = meta.get("shape", ())
    if shape == ():
        return 1
    out = 1
    for dim in shape:
        out *= int(dim)
    return int(out)


def estimate_csc_npz_load_bytes(path: Path) -> int:
    """Conservative peak estimate for reading this dependency-light CSC loader."""
    with zipfile.ZipFile(path, "r") as zf:
        metas = {}
        raw_bytes = 0
        for name in ("indices.npy", "indptr.npy", "shape.npy", "format.npy"):
            meta, file_size = read_npy_meta_from_zip(zf, name)
            metas[name] = meta
            raw_bytes += int(file_size)
        # data.npy is not loaded by the merge path; it is regenerated as all-ones on write.
        data_meta, _data_size = read_npy_meta_from_zip(zf, "data.npy")
        metas["data.npy"] = data_meta
    indices_len = array_len_from_meta(metas["indices.npy"])
    indptr_len = array_len_from_meta(metas["indptr.npy"])
    shape_len = array_len_from_meta(metas["shape.npy"])
    py_int_bytes = (indices_len + indptr_len + shape_len) * 36
    return int((raw_bytes + py_int_bytes) * 1.25 + 64 * 1024**2)


def decode_int_array(meta: dict, payload: bytes):
    descr = meta["descr"]
    itemsize = dtype_itemsize(descr)
    signed = "i" in descr
    if len(payload) % itemsize:
        raise ValueError(f"payload size is not divisible by dtype itemsize: {descr}")

    typecode = {
        (1, True): "b",
        (1, False): "B",
        (2, True): "h",
        (2, False): "H",
        (4, True): "i",
        (4, False): "I",
        (8, True): "q",
        (8, False): "Q",
    }.get((itemsize, signed))
    if typecode is None:
        endian = "big" if descr.startswith(">") else "little"
        return [
            int.from_bytes(payload[i:i + itemsize], endian, signed=signed)
            for i in range(0, len(payload), itemsize)
        ]

    arr = array(typecode)
    arr.frombytes(payload)
    if descr.startswith(">") and sys.byteorder != "big":
        arr.byteswap()
    elif descr.startswith("<") and sys.byteorder != "little":
        arr.byteswap()
    return arr


def read_csc_npz(path: Path, memory_limit_bytes: int = 0) -> dict:
    check_memory_budget(
        f"read {path}",
        estimate_csc_npz_load_bytes(path),
        memory_limit_bytes,
    )
    with zipfile.ZipFile(path, "r") as zf:
        arrays = {}
        for name in ("indices.npy", "indptr.npy", "shape.npy", "format.npy"):
            meta, payload = parse_npy(zf.read(name))
            arrays[name[:-4]] = (meta, payload)
        data_meta, _data_size = read_npy_meta_from_zip(zf, "data.npy")

    fmt_payload = arrays["format"][1]
    fmt = fmt_payload.rstrip(b"\x00").decode("ascii", errors="ignore")
    if fmt != "csc":
        raise ValueError(f"{path} is {fmt!r}, expected csc")

    shape = decode_int_array(*arrays["shape"])
    if len(shape) != 2:
        raise ValueError(f"{path} has bad shape array: {shape}")
    n_rows = int(shape[0])
    n_cols = int(shape[1])
    indices = decode_int_array(*arrays["indices"])
    indptr = decode_int_array(*arrays["indptr"])
    if array_len_from_meta(data_meta) != len(indices):
        raise ValueError(
            f"{path}: data length {array_len_from_meta(data_meta)} != len(indices)={len(indices)}; "
            f"NPZ 内部数据不一致，请重新导出"
        )

    # 输入校验：防止 indptr 长度不匹配导致的 IndexError
    if len(indptr) != n_cols + 1:
        raise ValueError(
            f"{path}: indptr length {len(indptr)} != n_cols + 1 ({n_cols + 1}); "
            f"NPZ 可能由包含重复信号的旧版导出器生成，请用修复后的 C++ 代码重新导出"
        )
    if indptr and indptr[-1] != len(indices):
        raise ValueError(
            f"{path}: indptr[-1]={indptr[-1]} != len(indices)={len(indices)}; "
            f"NPZ 内部数据不一致，请重新导出"
        )
    if indices and (min(indices) < 0 or max(indices) >= n_rows):
        raise ValueError(
            f"{path}: indices 范围 [{min(indices)}, {max(indices)}] 超出 [0, {n_rows}); "
            f"NPZ 数据异常，请重新导出"
        )

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "indices": indices,
        "indptr": indptr,
    }


def read_csc_npz_stats(path: Path) -> tuple[int, int, int]:
    with zipfile.ZipFile(path, "r") as zf:
        shape_meta, shape_payload = parse_npy(zf.read("shape.npy"))
        data_meta, _data_size = read_npy_meta_from_zip(zf, "data.npy")
        fmt_meta, fmt_payload = parse_npy(zf.read("format.npy"))

    _ = fmt_meta
    fmt = fmt_payload.rstrip(b"\x00").decode("ascii", errors="ignore")
    if fmt != "csc":
        raise ValueError(f"{path} is {fmt!r}, expected csc")
    shape = decode_int_array(shape_meta, shape_payload)
    if len(shape) != 2:
        raise ValueError(f"{path} has bad shape array: {shape}")
    return int(shape[0]), int(shape[1]), int(array_len_from_meta(data_meta))


def read_csc_ref_stats(file_ref) -> tuple[int, int, int]:
    if isinstance(file_ref, (str, Path)):
        return read_csc_npz_stats(Path(file_ref))

    total_rows = 0
    total_nnz = 0
    n_cols: int | None = None
    for p in file_ref:
        rows, cols, nnz = read_csc_npz_stats(Path(p))
        total_rows += int(rows)
        total_nnz += int(nnz)
        if n_cols is None:
            n_cols = int(cols)
        elif n_cols != int(cols):
            raise ValueError(f"row-shard n_cols mismatch while reading existing output: {p}")
    return int(total_rows), int(n_cols or 0), int(total_nnz)


def read_csc_npz_maybe_sharded(file_ref, memory_limit_bytes: int = 0) -> dict:
    """Load CSC matrix from a single NPZ path (str/Path) or a list of row-shard paths.

    When file_ref is a list, each shard is loaded independently and the row
    indices are adjusted so the result is a single vstack-ed CSC matrix.
    """
    if isinstance(file_ref, (str, Path)):
        return read_csc_npz(Path(file_ref), memory_limit_bytes=memory_limit_bytes)

    # Row-sharded: load each shard and vstack by columns
    shard_est = sum(estimate_csc_npz_load_bytes(Path(p)) for p in file_ref)
    check_memory_budget("read row-sharded source", shard_est, memory_limit_bytes)
    shards = [read_csc_npz(Path(p), memory_limit_bytes=memory_limit_bytes) for p in file_ref]
    if len(shards) == 1:
        return shards[0]

    n_cols = shards[0]["n_cols"]
    for s in shards[1:]:
        if s["n_cols"] != n_cols:
            raise ValueError("row-shard n_cols mismatch")

    # For CSC row-vstack: for each column, iterate shards in order,
    # appending each shard's indices shifted by its cumulative row offset.
    merged_indices: list[int] = []
    merged_indptr: list[int] = [0]
    total_rows = sum(s["n_rows"] for s in shards)
    total_nnz = sum(len(s["indices"]) for s in shards)
    merged_est = (total_nnz + n_cols + 1) * 36 + 64 * 1024**2
    check_memory_budget("merge row-sharded source in memory", merged_est, memory_limit_bytes)

    # Precompute per-shard row base offset
    row_bases = [0]
    for s in shards[:-1]:
        row_bases.append(row_bases[-1] + s["n_rows"])

    for col in range(n_cols):
        for si, s in enumerate(shards):
            start, end = s["indptr"][col], s["indptr"][col + 1]
            base = int(row_bases[si])
            for k in range(start, end):
                merged_indices.append(int(s["indices"][k]) + base)
        merged_indptr.append(len(merged_indices))

    return {
        "n_rows": int(total_rows),
        "n_cols": int(n_cols),
        "indices": merged_indices,
        "indptr": merged_indptr,
    }


def npy_header(descr: str, shape: tuple[int, ...]) -> bytes:
    shape_str = "(" + ", ".join(str(x) for x in shape)
    if len(shape) == 1:
        shape_str += ","
    shape_str += ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}"
    preamble = 10
    pad = 16 - ((preamble + len(header) + 1) % 16)
    if pad == 16:
        pad = 0
    header = header + (" " * pad) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1")


def int_payload(values: list[int], itemsize: int, signed: bool) -> bytes:
    if itemsize == 4 and signed:
        arr = array("i", values)
    elif itemsize == 8 and signed:
        arr = array("q", values)
    elif itemsize == 4 and not signed:
        arr = array("I", values)
    elif itemsize == 8 and not signed:
        arr = array("Q", values)
    else:
        return b"".join(int(v).to_bytes(itemsize, "little", signed=signed) for v in values)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def make_npy(descr: str, shape: tuple[int, ...], payload: bytes) -> bytes:
    return npy_header(descr, shape) + payload


def write_csc_npz(path: Path, n_rows: int, n_cols: int, indices: list[int], indptr: list[int]) -> None:
    mkdirp(path.parent)
    if len(indptr) != n_cols + 1:
        raise ValueError("bad output indptr length")
    if any(i < 0 or i > 0x7fffffff for i in indices):
        idx_descr = "<i8"
        idx_payload = int_payload(indices, 8, True)
    else:
        idx_descr = "<i4"
        idx_payload = int_payload(indices, 4, True)

    arrays = {
        "data.npy": make_npy("|u1", (len(indices),), b"\x01" * len(indices)),
        "indices.npy": make_npy(idx_descr, (len(indices),), idx_payload),
        "indptr.npy": make_npy("<i8", (len(indptr),), int_payload(indptr, 8, True)),
        "shape.npy": make_npy("<i8", (2,), int_payload([n_rows, n_cols], 8, True)),
        "format.npy": make_npy("|S3", (), b"csc"),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in arrays.items():
            zf.writestr(name, payload)


# Standard ZIP single-file limit (4 GiB minus safety margin)
_ZIP_SAFE_BYTES = 0xFFFF0000


def _total_npz_bytes_est(nnz: int, n_cols: int) -> int:
    """Estimate total .npz file size for a CSC matrix with given nnz and n_cols."""
    idx_sz = nnz * 4          # indices  (int32)
    data_sz = nnz             # data     (uint8)
    ptr_sz = (n_cols + 1) * 8  # indptr   (int64)
    overhead = 5 * 400         # NPY headers + ZIP local/central dir
    return idx_sz + data_sz + ptr_sz + overhead


def estimate_python_csc_bytes(nnz: int, n_cols: int) -> int:
    # Output assembly still uses Python lists before bulk serializing to NPY.
    return int((int(nnz) + int(n_cols) + 1) * 36 + 64 * 1024**2)


def auto_row_block_rows(n_rows: int, nnz: int, n_cols: int,
                        memory_limit_bytes: int,
                        max_cycles_per_file: int = 0) -> int:
    rows = int(n_rows) if n_rows > 0 else 1
    if max_cycles_per_file > 0:
        rows = min(rows, int(max_cycles_per_file))

    nnz_per_row = float(nnz) / max(1.0, float(n_rows))
    bytes_per_nnz = 5  # int32 indices + uint8 data in the output NPZ
    if nnz_per_row > 0:
        zip_rows = max(1, int(float(_ZIP_SAFE_BYTES) / (nnz_per_row * bytes_per_nnz)))
        rows = min(rows, zip_rows)

    if memory_limit_bytes > 0 and nnz_per_row > 0:
        shard_budget = max(1, int(memory_limit_bytes * 0.70))
        mem_rows = max(1, int(shard_budget / max(1.0, nnz_per_row * 36.0)))
        rows = min(rows, mem_rows)

    return max(1, min(rows, int(n_rows) if n_rows > 0 else 1))


def extend_shifted_rows_in_range(out: list[int], rows, start: int, end: int,
                                 row_start: int, row_end: int) -> None:
    # CSC cache columns are written in increasing row order, so bisect avoids
    # re-scanning an entire column for every row shard.
    lo = bisect_left(rows, row_start, start, end)
    hi = bisect_left(rows, row_end, lo, end)
    out.extend(int(r) - row_start for r in rows[lo:hi])


def subset_csc_and_write(matrix: dict, src_names: list[str], dst_names: list[str],
                         dst_npz: Path, memory_limit_bytes: int = 0,
                         max_cycles_per_file: int = 0) -> tuple[list[str], int]:
    if matrix["n_cols"] != len(src_names):
        raise ValueError(f"{dst_npz}: matrix n_cols={matrix['n_cols']} but names={len(src_names)}")

    name_to_col = {name: i for i, name in enumerate(src_names)}
    missing = [name for name in dst_names if name not in name_to_col]
    if missing:
        raise ValueError(f"{dst_npz}: missing clean columns: {missing[:5]}")

    selected_cols = [name_to_col[name] for name in dst_names]
    src_indices = matrix["indices"]
    src_indptr = matrix["indptr"]
    n_rows = int(matrix["n_rows"])
    n_cols = len(dst_names)
    nnz = sum(int(src_indptr[col + 1]) - int(src_indptr[col]) for col in selected_cols)
    single_npz_safe = _total_npz_bytes_est(nnz, n_cols) <= _ZIP_SAFE_BYTES
    single_mem_safe = memory_limit_bytes <= 0 or estimate_python_csc_bytes(nnz, n_cols) <= memory_limit_bytes

    if single_npz_safe and single_mem_safe and not (max_cycles_per_file > 0 and n_rows > max_cycles_per_file):
        out_indices: list[int] = []
        out_indptr: list[int] = [0]
        for col in selected_cols:
            start, end = src_indptr[col], src_indptr[col + 1]
            out_indices.extend(src_indices[start:end])
            out_indptr.append(len(out_indices))
        write_csc_npz(dst_npz, n_rows, n_cols, out_indices, out_indptr)
        paths = [str(dst_npz)]
        del out_indices, out_indptr
        gc.collect()
        return paths, int(nnz)

    max_rows = auto_row_block_rows(n_rows, nnz, n_cols, memory_limit_bytes, max_cycles_per_file)
    base_str = str(dst_npz)
    if base_str.endswith(".npz"):
        base_str = base_str[:-4]
    paths: list[str] = []
    print(
        f"  merge memory/zip guard: {dst_npz.name} nnz={nnz} rows={n_rows} "
        f"cols={n_cols}; row-sharding with row_block_rows={max_rows}",
        flush=True,
    )
    for si, r0 in enumerate(range(0, n_rows, max_rows)):
        r1 = min(n_rows, r0 + max_rows)
        shard_indices: list[int] = []
        shard_indptr: list[int] = [0]
        for col in selected_cols:
            start, end = src_indptr[col], src_indptr[col + 1]
            extend_shifted_rows_in_range(shard_indices, src_indices, start, end, r0, r1)
            shard_indptr.append(len(shard_indices))
        check_memory_budget(
            f"assemble shard {Path(base_str).name}.row{si:05d}.npz",
            estimate_python_csc_bytes(len(shard_indices), n_cols),
            memory_limit_bytes,
        )
        shard_path = Path(f"{base_str}.row{si:05d}.npz")
        write_csc_npz(shard_path, r1 - r0, n_cols, shard_indices, shard_indptr)
        paths.append(str(shard_path))
        print(f"  merge row-shard {si}: rows={r1 - r0} nnz={len(shard_indices)} -> {shard_path.name}", flush=True)
        del shard_indices, shard_indptr
        gc.collect()

    return paths, int(nnz)

def read_uint8_npy(path: Path) -> list[int]:
    meta, payload = parse_npy(path.read_bytes())
    descr = str(meta.get("descr", ""))
    shape = tuple(meta.get("shape", ()))
    if descr not in ("|u1", "<u1", ">u1"):
        raise ValueError(f"{path}: expected uint8 npy, got descr={descr!r}")
    if len(shape) != 1:
        raise ValueError(f"{path}: expected 1-D initial level vector, got shape={shape}")
    if len(payload) != int(shape[0]):
        raise ValueError(f"{path}: payload bytes={len(payload)} != shape[0]={shape[0]}")
    return list(payload)


def write_uint8_npy(path: Path, values: list[int]) -> None:
    mkdirp(path.parent)
    payload = bytes(1 if int(v) else 0 for v in values)
    path.write_bytes(make_npy("|u1", (len(values),), payload))


def initial_vector_from_full_level_csc(matrix: dict, src_names: list[str]) -> list[int]:
    if matrix["n_cols"] != len(src_names):
        raise ValueError(f"full level matrix n_cols={matrix['n_cols']} but names={len(src_names)}")
    src_indices = matrix["indices"]
    src_indptr = matrix["indptr"]
    values: list[int] = []
    for col in range(len(src_names)):
        start, end = int(src_indptr[col]), int(src_indptr[col + 1])
        pos = bisect_left(src_indices, 0, start, end)
        values.append(1 if pos < end and int(src_indices[pos]) == 0 else 0)
    return values


def write_initial_level_npy(initial_values: list[int], src_names: list[str], dst_names: list[str],
                            dst_npy: Path) -> tuple[str, int]:
    """Write row-0 level values as a uint8 .npy vector after column reordering."""
    if len(initial_values) != len(src_names):
        raise ValueError(f"{dst_npy}: initial vector len={len(initial_values)} but names={len(src_names)}")

    name_to_col = {name: i for i, name in enumerate(src_names)}
    missing = [name for name in dst_names if name not in name_to_col]
    if missing:
        raise ValueError(f"{dst_npy}: missing clean columns: {missing[:5]}")

    out = [1 if int(initial_values[name_to_col[name]]) else 0 for name in dst_names]
    write_uint8_npy(dst_npy, out)
    nnz = int(sum(out))
    del out
    gc.collect()
    return str(dst_npy), nnz

def unique_marker_name(base: str, used: set[str]) -> str:
    candidate = base
    i = 1
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def read_original_headers(manifest_path: Path) -> list[str] | None:
    p = manifest_path.parent.parent / "original_headers.txt"
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


_BLOCK_OUTPUT_RE = re.compile(r"\.block(\d+)\.")


def scan_latest_output_block_id(output_dir: Path) -> int | None:
    if not output_dir.is_dir():
        return None
    latest: int | None = None
    for p in output_dir.rglob("*.npz"):
        m = _BLOCK_OUTPUT_RE.search(p.name)
        if not m:
            continue
        block_id = int(m.group(1))
        if latest is None or block_id > latest:
            latest = block_id
    return latest


def existing_npz_ref(base_npz: Path):
    if base_npz.is_file():
        return str(base_npz)
    base = str(base_npz)
    if base.endswith(".npz"):
        base = base[:-4]
    shards = sorted(base_npz.parent.glob(Path(base).name + ".row*.npz"))
    if shards:
        return [str(p) for p in shards]
    return None


def remove_npz_ref(base_npz: Path) -> None:
    if base_npz.is_file():
        base_npz.unlink()
    base = str(base_npz)
    if base.endswith(".npz"):
        base = base[:-4]
    for shard in base_npz.parent.glob(Path(base).name + ".row*.npz"):
        shard.unlink()


def block_checkpoint_path(checkpoint_dir: Path, block_id: int) -> Path:
    return checkpoint_dir / f"block{int(block_id):05d}.json"


def load_block_checkpoint(checkpoint_dir: Path, block_id: int) -> OrderedDict | None:
    p = block_checkpoint_path(checkpoint_dir, block_id)
    if not p.is_file():
        return None
    block = load_json(p)
    if int(block.get("block_id", -1)) != int(block_id):
        raise ValueError(f"bad block checkpoint block_id in {p}")
    return block


def write_block_checkpoint(checkpoint_dir: Path, out_block: OrderedDict) -> None:
    mkdirp(checkpoint_dir)
    p = block_checkpoint_path(checkpoint_dir, int(out_block["block_id"]))
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(out_block, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def clean_names_for_block(bi: int, manifests: list[OrderedDict], layouts: list[list[dict]],
                          manifest_paths: list[Path]) -> tuple[list[str], list[str]]:
    ordered_by_manifest = [
        ordered_names_for_block(manifests[mi]["blocks"][bi], layouts[mi])
        for mi in range(len(manifests))
    ]
    base_order = ordered_by_manifest[0]
    for mi, order in enumerate(ordered_by_manifest[1:], 1):
        if order != base_order:
            raise ValueError(
                f"block {bi + 1} layout mismatch between {manifest_paths[0]} and {manifest_paths[mi]}"
            )

    dirty = set()
    for manifest in manifests:
        dirty.update(manifest["blocks"][bi].get("removed_xz_names", []))
    clean_names = [name for name in base_order if name not in dirty]
    removed_xz = [name for name in base_order if name in dirty]
    return clean_names, removed_xz


def make_out_block(base_block: dict, clean_names: list[str], removed_xz: list[str]) -> OrderedDict:
    out_block = OrderedDict()
    out_block["block_id"] = int(base_block["block_id"])
    out_block["raw_indices"] = list(base_block["raw_indices"])
    out_block["names"] = clean_names
    out_block["removed_xz_names"] = removed_xz
    out_block["files"] = OrderedDict()
    out_block["initial_level_files"] = OrderedDict()
    out_block["n_rows"] = OrderedDict()
    out_block["level_n_rows"] = OrderedDict()
    out_block["nnz"] = OrderedDict()
    out_block["initial_level_nnz"] = OrderedDict()
    return out_block


def tog_output_path(cache_dir: Path, marker: str, block_id: int) -> Path:
    return cache_dir / f"{safe_stem(marker)}.block{int(block_id):05d}.tog.cache.csc.npz"


def initial_level_output_path(cache_dir: Path, marker: str, block_id: int) -> Path:
    return cache_dir / f"{safe_stem(marker)}.block{int(block_id):05d}.initial_level.cache.uint8.npy"


def remove_block_outputs(cache_dir: Path, marker_map: dict[tuple[int, str], str],
                         manifests: list[OrderedDict], bi: int, block_id: int) -> None:
    for mi, manifest in enumerate(manifests):
        src_block = manifest["blocks"][bi]
        for csv_path in manifest["csv_paths"]:
            marker = marker_map[(mi, csv_path)]
            remove_npz_ref(tog_output_path(cache_dir, marker, block_id))
            if (csv_path in src_block.get("level_files", {})
                    or csv_path in src_block.get("initial_level_files", {})):
                remove_npz_ref(initial_level_output_path(cache_dir, marker, block_id))


def reconstruct_existing_out_block(base_block: dict, clean_names: list[str], removed_xz: list[str],
                                   manifests: list[OrderedDict], marker_map: dict[tuple[int, str], str],
                                   cache_dir: Path, bi: int) -> OrderedDict:
    block_id = int(base_block["block_id"])
    out_block = make_out_block(base_block, clean_names, removed_xz)
    for mi, manifest in enumerate(manifests):
        src_block = manifest["blocks"][bi]
        for csv_path in manifest["csv_paths"]:
            marker = marker_map[(mi, csv_path)]
            out_npz = tog_output_path(cache_dir, marker, block_id)
            out_ref = existing_npz_ref(out_npz)
            if out_ref is None:
                raise FileNotFoundError(f"missing existing output for resume: {out_npz}")
            rows, cols, nnz = read_csc_ref_stats(out_ref)
            if cols != len(clean_names):
                raise ValueError(
                    f"{out_npz}: existing output has n_cols={cols}, expected clean names={len(clean_names)}"
                )
            out_block["files"][marker] = out_ref
            out_block["n_rows"][marker] = int(rows)
            out_block["nnz"][marker] = int(nnz)

            if (csv_path in src_block.get("level_files", {})
                    or csv_path in src_block.get("initial_level_files", {})):
                level_npy = initial_level_output_path(cache_dir, marker, block_id)
                level_ref = existing_npz_ref(level_npy)
                if level_ref is None:
                    raise FileNotFoundError(f"missing existing initial level output for resume: {level_npy}")
                initial_values = read_uint8_npy(Path(level_ref))
                if len(initial_values) != len(clean_names):
                    raise ValueError(
                        f"{level_npy}: existing initial level vector len={len(initial_values)}, "
                        f"expected clean names={len(clean_names)}"
                    )
                out_block["initial_level_files"][marker] = level_ref
                out_block["level_n_rows"][marker] = int(src_block.get("level_n_rows", {}).get(csv_path, out_block["n_rows"][marker] + 1))
                out_block["initial_level_nnz"][marker] = int(sum(initial_values))
    return out_block


def prepare_merge_context(manifest_paths: list[Path], output_dir: Path) -> dict:
    if len(manifest_paths) < 2:
        raise ValueError("need at least two manifests to merge")
    mkdirp(output_dir)
    work_dir = output_dir / "_work"
    cache_dir = work_dir / "tog_cache"
    marker_dir = work_dir / "source_markers"
    checkpoint_dir = work_dir / "tog_cache_block_manifests"
    mkdirp(cache_dir)
    mkdirp(marker_dir)
    mkdirp(checkpoint_dir)

    manifests = [load_json(p) for p in manifest_paths]
    layouts = [load_layout(p) for p in manifest_paths]
    if any(m.get("cache_format") != "csc" for m in manifests):
        raise ValueError("all input manifests must be CSC cache manifests")

    base_blocks = manifests[0]["blocks"]
    if any(len(m["blocks"]) != len(base_blocks) for m in manifests):
        raise ValueError("all manifests must use the same block partition")

    headers = read_original_headers(manifest_paths[0])
    if headers:
        for p in manifest_paths[1:]:
            other = read_original_headers(p)
            if other and other != headers:
                raise ValueError(f"original_headers mismatch: {p}")
        with (output_dir / "original_headers.txt").open("w", encoding="utf-8") as f:
            for h in headers:
                f.write(h + "\n")

    shutil.copyfile(manifest_paths[0].parent / "expanded_layout.jsonl", work_dir / "expanded_layout.jsonl")

    marker_map: dict[tuple[int, str], str] = {}
    source_map: list[OrderedDict] = []
    used_markers: set[str] = set()
    merged_csv_paths: list[str] = []
    for mi, manifest in enumerate(manifests):
        csv_paths = list(manifest.get("csv_paths", []))
        if not csv_paths:
            raise ValueError(f"input manifest has empty csv_paths: {manifest_paths[mi]}")
        for csv_path in csv_paths:
            base = unique_marker_name(f"m{mi:03d}_{safe_stem(csv_path)}", used_markers)
            marker = marker_dir / f"{base}.fsdbtog"
            source_fsdb = manifest.get('source_fsdb', '')
            marker.write_text(
                f"source_manifest_index={mi}\n"
                f"source_manifest={manifest_paths[mi]}\n"
                f"source_csv_path={csv_path}\n"
                f"source_fsdb={source_fsdb}\n",
                encoding="utf-8",
            )
            marker_s = str(marker)
            marker_map[(mi, csv_path)] = marker_s
            merged_csv_paths.append(marker_s)
            rec = OrderedDict()
            rec["manifest_index"] = int(mi)
            rec["input_manifest"] = str(manifest_paths[mi])
            rec["source_csv_path"] = str(csv_path)
            rec["source_fsdb"] = str(source_fsdb)
            rec["output_marker"] = marker_s
            source_map.append(rec)

    return {
        "manifest_paths": manifest_paths,
        "output_dir": output_dir,
        "work_dir": work_dir,
        "cache_dir": cache_dir,
        "checkpoint_dir": checkpoint_dir,
        "manifests": manifests,
        "layouts": layouts,
        "base_blocks": base_blocks,
        "marker_map": marker_map,
        "source_map": source_map,
        "merged_csv_paths": merged_csv_paths,
    }


def process_merge_block(ctx: dict, bi: int, memory_limit_bytes: int = 0,
                        max_cycles_per_file: int = 0,
                        resume: bool = True,
                        rebuild: bool = False) -> OrderedDict:
    manifests = ctx["manifests"]
    layouts = ctx["layouts"]
    manifest_paths = ctx["manifest_paths"]
    base_blocks = ctx["base_blocks"]
    marker_map = ctx["marker_map"]
    cache_dir = ctx["cache_dir"]
    checkpoint_dir = ctx["checkpoint_dir"]
    merged_csv_paths = ctx["merged_csv_paths"]

    base_block = base_blocks[bi]
    block_id = int(base_block["block_id"])
    clean_names, removed_xz = clean_names_for_block(bi, manifests, layouts, manifest_paths)

    if resume and not rebuild:
        out_block = load_block_checkpoint(checkpoint_dir, block_id)
        if out_block is not None:
            print(
                f"resume skip block {bi + 1}/{len(base_blocks)}: block_id={block_id} "
                f"clean={len(clean_names)} removed_xz={len(removed_xz)}",
                flush=True,
            )
            return out_block

    if rebuild:
        remove_block_outputs(cache_dir, marker_map, manifests, bi, block_id)
        checkpoint = block_checkpoint_path(checkpoint_dir, block_id)
        if checkpoint.is_file():
            checkpoint.unlink()

    out_block = make_out_block(base_block, clean_names, removed_xz)

    for mi, manifest in enumerate(manifests):
        src_block = manifest["blocks"][bi]
        src_names = list(src_block["names"])
        for csv_path in manifest["csv_paths"]:
            marker = marker_map[(mi, csv_path)]
            out_npz = tog_output_path(cache_dir, marker, block_id)

            src_matrix = read_csc_npz_maybe_sharded(
                src_block["files"][csv_path],
                memory_limit_bytes=memory_limit_bytes,
            )
            src_names_ordered = src_names

            out_paths, out_nnz = subset_csc_and_write(
                src_matrix, src_names_ordered, clean_names, out_npz,
                memory_limit_bytes=memory_limit_bytes,
                max_cycles_per_file=max_cycles_per_file,
            )
            out_block["files"][marker] = out_paths if len(out_paths) > 1 else out_paths[0]
            out_block["n_rows"][marker] = int(src_matrix["n_rows"])
            out_block["nnz"][marker] = int(out_nnz)

            initial_values = None
            level_rows = None
            if csv_path in src_block.get("initial_level_files", {}):
                initial_values = read_uint8_npy(Path(src_block["initial_level_files"][csv_path]))
                level_rows = int(src_block.get("level_n_rows", {}).get(csv_path, src_matrix["n_rows"] + 1))
            elif csv_path in src_block.get("level_files", {}):
                level_matrix = read_csc_npz_maybe_sharded(
                    src_block["level_files"][csv_path],
                    memory_limit_bytes=memory_limit_bytes,
                )
                initial_values = initial_vector_from_full_level_csc(level_matrix, src_names_ordered)
                level_rows = int(level_matrix["n_rows"])
                del level_matrix
                gc.collect()

            if initial_values is not None:
                initial_level_npy = initial_level_output_path(cache_dir, marker, block_id)
                initial_path, initial_nnz = write_initial_level_npy(
                    initial_values, src_names_ordered, clean_names, initial_level_npy,
                )
                out_block["initial_level_files"][marker] = initial_path
                out_block["level_n_rows"][marker] = int(level_rows)
                out_block["initial_level_nnz"][marker] = int(initial_nnz)

            del src_matrix
            gc.collect()

    write_block_checkpoint(checkpoint_dir, out_block)
    print(
        f"block {bi + 1}/{len(base_blocks)}: block_id={block_id} clean={len(clean_names)} "
        f"removed_xz={len(removed_xz)} files={len(merged_csv_paths)}",
        flush=True,
    )
    return out_block


def finalize_merged_manifest(ctx: dict) -> Path:
    manifests = ctx["manifests"]
    manifest_paths = ctx["manifest_paths"]
    layouts = ctx["layouts"]
    base_blocks = ctx["base_blocks"]
    checkpoint_dir = ctx["checkpoint_dir"]

    out_blocks = []
    clean_names_after_xz: list[str] = []
    xz_removed_count = 0
    for bi, base_block in enumerate(base_blocks):
        block_id = int(base_block["block_id"])
        clean_names, removed_xz = clean_names_for_block(bi, manifests, layouts, manifest_paths)
        clean_names_after_xz.extend(clean_names)
        xz_removed_count += len(removed_xz)
        out_block = load_block_checkpoint(checkpoint_dir, block_id)
        if out_block is None:
            raise FileNotFoundError(
                f"missing block checkpoint for block_id={block_id}: {block_checkpoint_path(checkpoint_dir, block_id)}"
            )
        out_blocks.append(out_block)

    merged = OrderedDict()
    merged["cache_format"] = "csc"
    merged["input_data_type"] = "tog"
    merged["tog_diff_applied"] = False
    merged["tog_note"] = "Merged FSDB-generated TOG cache; x/z removal uses union across all input manifests."
    merged["level_cache_available"] = any(
        any(block.get("level_files") or block.get("initial_level_files") for block in manifest.get("blocks", []))
        for manifest in manifests
    )
    merged["level_cache_mode"] = "initial_plus_tog"
    merged["level_note"] = "Merged level cache is stored as one uint8 .npy initial level vector per case/block plus full TOG; reduction restores full level after column cleaning."
    merged["xz_policy"] = "sampled-level-only true X/Z dirty; NC holds previous; NV/no-value defaults to 0; merged by union across all input manifests"
    merged["input_manifests"] = [str(p) for p in manifest_paths]
    merged["source_map"] = ctx["source_map"]
    merged["blocks"] = out_blocks
    merged["csv_paths"] = ctx["merged_csv_paths"]
    merged["clean_names_after_xz"] = clean_names_after_xz
    merged["xz_removed_count"] = int(xz_removed_count)
    merged["num_blocks"] = len(out_blocks)

    out_manifest = ctx["work_dir"] / "tog_cache_manifest.json"
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return out_manifest


def merge_manifests(manifest_paths: list[Path], output_dir: Path,
                    memory_limit_bytes: int = 0,
                    max_cycles_per_file: int = 0,
                    resume: bool = True,
                    only_block_id: int | None = None,
                    finalize_only: bool = False) -> Path | None:
    ctx = prepare_merge_context(manifest_paths, output_dir)
    base_blocks = ctx["base_blocks"]

    if finalize_only:
        return finalize_merged_manifest(ctx)

    if only_block_id is not None:
        block_id_to_index = {int(block["block_id"]): i for i, block in enumerate(base_blocks)}
        if int(only_block_id) not in block_id_to_index:
            raise ValueError(f"block_id={only_block_id} is not present in the input manifests")
        process_merge_block(
            ctx,
            block_id_to_index[int(only_block_id)],
            memory_limit_bytes=memory_limit_bytes,
            max_cycles_per_file=max_cycles_per_file,
            resume=resume,
            rebuild=not resume,
        )
        return block_checkpoint_path(ctx["checkpoint_dir"], int(only_block_id))

    block_id_to_index = {int(block["block_id"]): i for i, block in enumerate(base_blocks)}
    resume_start_index = 0
    latest_output_block_id = scan_latest_output_block_id(ctx["cache_dir"]) if resume else None
    if latest_output_block_id is not None:
        if latest_output_block_id not in block_id_to_index:
            print(
                f"resume: ignore existing output block_id={latest_output_block_id}; "
                "it is not present in the input manifests",
                flush=True,
            )
        else:
            resume_start_index = int(block_id_to_index[latest_output_block_id])
            print(
                f"resume: latest existing output block_id={latest_output_block_id}; "
                f"restart from block {resume_start_index + 1}/{len(base_blocks)}",
                flush=True,
            )

    for bi, base_block in enumerate(base_blocks):
        block_id = int(base_block["block_id"])
        clean_names, removed_xz = clean_names_for_block(bi, ctx["manifests"], ctx["layouts"], ctx["manifest_paths"])
        if resume and bi < resume_start_index:
            out_block = load_block_checkpoint(ctx["checkpoint_dir"], block_id)
            if out_block is None:
                out_block = reconstruct_existing_out_block(
                    base_block, clean_names, removed_xz,
                    ctx["manifests"], ctx["marker_map"], ctx["cache_dir"], bi,
                )
                write_block_checkpoint(ctx["checkpoint_dir"], out_block)
            print(
                f"resume skip block {bi + 1}/{len(base_blocks)}: block_id={block_id} "
                f"clean={len(clean_names)} removed_xz={len(removed_xz)}",
                flush=True,
            )
            continue

        process_merge_block(
            ctx,
            bi,
            memory_limit_bytes=memory_limit_bytes,
            max_cycles_per_file=max_cycles_per_file,
            resume=False,
            rebuild=resume and latest_output_block_id is not None and bi == resume_start_index,
        )

    return finalize_merged_manifest(ctx)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-FSDB TOG cache manifests for global reduction.")
    p.add_argument("--manifest", action="append", required=True, help="Input manifest. Repeat for each FSDB.")
    p.add_argument("--output-dir", required=True, help="Merged output directory.")
    p.add_argument("--max-mem", default="28GB", help="merge memory budget, e.g. 28GB, 32000MB; 0 disables checks")
    p.add_argument("--merge-mem-frac", type=float, default=0.60,
                   help="fraction of --max-mem allowed for loading/assembling one merge matrix")
    p.add_argument("--max-cycles-per-file", type=int, default=0,
                   help="force row-sharded output when rows exceed this value; 0 = auto only")
    p.add_argument("--no-resume", action="store_true",
                   help="disable automatic block-level resume and rebuild all blocks")
    p.add_argument("--only-block-id", type=int, default=None,
                   help="merge only this block_id and write its block checkpoint; do not finalize the full manifest")
    p.add_argument("--finalize-only", action="store_true",
                   help="only assemble the final manifest from block checkpoints written by prior --only-block-id runs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_paths = [Path(p).resolve() for p in args.manifest]
    max_mem_bytes = parse_mem_to_bytes(args.max_mem)
    memory_limit_bytes = int(max_mem_bytes * float(args.merge_mem_frac)) if max_mem_bytes > 0 else 0
    out_manifest = merge_manifests(
        manifest_paths,
        Path(args.output_dir).resolve(),
        memory_limit_bytes=memory_limit_bytes,
        max_cycles_per_file=args.max_cycles_per_file,
        resume=not args.no_resume,
        only_block_id=args.only_block_id,
        finalize_only=args.finalize_only,
    )
    print(f"merged manifest: {out_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
