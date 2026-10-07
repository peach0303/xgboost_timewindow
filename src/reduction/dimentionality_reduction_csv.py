#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large-column CSV(already TOG) -> TOG cache -> column cleaning/reduction pipeline.

Final design used here:
1. CSV is read only once for generating TOG cache.
2. First column is treated as time and dropped.
3. Multi-bit full-width CSV cells such as 00001xz0 are split to bit columns.
4. x/z/X/Z in a raw signal invalidates all split bit columns of that raw signal.
5. Input CSV is assumed to already contain TOG data, so no level->tog XOR diff is applied.
   All valid rows are preserved.
6. After x/z cleaning, TOG data is cached as column-block sparse NPZ, default CSC.
7. Later stages use only the cached NPZ files, not the original CSV files:
   - remove all-zero columns
   - merge all-one columns, keeping one representative
   - exact duplicate removal across all CSV files jointly
   - optional MinHash/LSH similarity reduction across all CSV files jointly
8. Stage/final snapshots are saved as one NPZ per input CSV when memory estimate allows.
   If the full per-CSV output is estimated to exceed the configured memory budget,
   snapshots are automatically split by row blocks; columns are never split in final snapshots.

Important trade-off:
- This avoids repeated CSV reading and controls RAM by column-block internal cache.
- Internal cache is column-blocked for column operations; final/stage outputs keep all columns together per CSV unless row-splitting is needed.
- It uses disk space for cache files.
"""

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz, load_npz


# -----------------------------
# CSV/TGZ file helpers
# -----------------------------
def _is_tgz(path):
    """Check if path is a .csv.tgz or .tgz file."""
    return str(path).endswith('.csv.tgz') or str(path).endswith('.tgz')


def open_csv_text(csv_path):
    """Open a CSV file (plain or .tgz) for text reading line-by-line.
    Returns (file_object, closer_callable) tuple; caller should call closer() when done.
    """
    if _is_tgz(csv_path):
        proc = subprocess.Popen(
            ['tar', 'xzf', str(csv_path), '-O'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        f = io.TextIOWrapper(proc.stdout, encoding='utf-8', newline='')
        def _close():
            try:
                f.close()
            except Exception:
                pass
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            proc.wait()
        return f, _close
    else:
        f = open(str(csv_path), 'r', encoding='utf-8', newline='')
        return f, f.close


def open_csv_binary(csv_path):
    """Open a CSV file (plain or .tgz) for binary reading line-by-line.
    Returns (file_object, closer_callable) tuple; caller should call closer() when done.
    """
    if _is_tgz(csv_path):
        proc = subprocess.Popen(
            ['tar', 'xzf', str(csv_path), '-O'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=64 * 1024 * 1024,
        )
        def _close():
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            proc.wait()
        return proc.stdout, _close
    else:
        f = open(str(csv_path), 'rb', buffering=64 * 1024 * 1024)
        return f, f.close


# -----------------------------
# basic helpers
# -----------------------------
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def mkdirp(p):
    if p:
        os.makedirs(p, exist_ok=True)


def write_lines(path, lines):
    mkdirp(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(str(x) + "\n")


def read_filelist(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not os.path.isfile(s):
                raise FileNotFoundError(f"filelist line {i}: file not found: {s}")
            out.append(s)
    if not out:
        raise ValueError("empty filelist")
    return out


def read_header(csv_path):
    f, closer = open_csv_text(csv_path)
    try:
        return next(csv.reader(f))
    finally:
        closer()


def ensure_same_header_streaming(csv_paths):
    if not csv_paths:
        raise ValueError("empty csv_paths")
    ref = read_header(csv_paths[0])
    log(f"HEADER reference: {csv_paths[0]}, cols={len(ref)}")
    for p in csv_paths[1:]:
        cur = read_header(p)
        if len(cur) != len(ref):
            raise ValueError(f"CSV header mismatch: {p}; ref_cols={len(ref)}, cur_cols={len(cur)}")
        if cur != ref:
            mismatch_idx = next((i for i, (a, b) in enumerate(zip(ref, cur)) if a != b), None)
            raise ValueError(
                f"CSV header mismatch: {p}; first_mismatch_col={mismatch_idx}; "
                f"ref={ref[mismatch_idx]!r}, cur={cur[mismatch_idx]!r}"
            )
        log(f"HEADER checked: {p}, cols={len(cur)}")
        del cur
        gc.collect()
    return ref


def parse_mem_to_bytes(s):
    s = str(s).strip().lower()
    m = re.fullmatch(r"([0-9.]+)\s*([kmgt]?b?)?", s)
    if not m:
        raise ValueError(f"bad memory size: {s}")
    val = float(m.group(1))
    unit = m.group(2) or "b"
    if unit in ("g", "gb"):
        return int(val * 1024**3)
    if unit in ("m", "mb"):
        return int(val * 1024**2)
    if unit in ("k", "kb"):
        return int(val * 1024)
    return int(val)


def safe_stem(path):
    return Path(path).stem.replace(os.sep, "_")


# -----------------------------
# column naming / bit expansion
# -----------------------------
_RANGE_RE = re.compile(r"^(.*)\[(-?\d+)\s*:\s*(-?\d+)\]$")


def bit_names_from_signal(signal_name, width):
    signal_name = str(signal_name)
    if width <= 1:
        return [signal_name]
    m = _RANGE_RE.match(signal_name)
    if m:
        base, left, right = m.group(1), int(m.group(2)), int(m.group(3))
        step = -1 if left >= right else 1
        bits = list(range(left, right + step, step))
        if len(bits) == width:
            return [f"{base}[{b}]" for b in bits]
    return [f"{signal_name}[{i}]" for i in range(width - 1, -1, -1)]


def detect_width_from_sample(value):
    if value is None:
        return 1
    s = str(value).strip()
    if len(s) > 1 and re.fullmatch(r"[01xXzZ]+", s):
        return len(s)
    return 1


def read_first_data_row(csv_path):
    f, closer = open_csv_text(csv_path)
    try:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                return row
    finally:
        closer()
    raise ValueError(f"no data row found: {csv_path}")


def build_expanded_layout(header, sample_row, output_jsonl):
    """Build expanded bit-column layout. raw_idx is CSV column index including time column."""
    records = []
    expanded_names = []
    raw_to_records = defaultdict(list)
    for raw_idx in range(1, len(header)):  # drop time column
        raw_name = header[raw_idx]
        sample = sample_row[raw_idx] if raw_idx < len(sample_row) else "0"
        width = detect_width_from_sample(sample)
        for bit_pos, bit_name in enumerate(bit_names_from_signal(raw_name, width)):
            rec = {
                "expanded_idx": len(records),
                "name": bit_name,
                "raw_idx": raw_idx,
                "raw_name": raw_name,
                "width": int(width),
                "bit_pos": int(bit_pos),
            }
            records.append(rec)
            expanded_names.append(bit_name)
            raw_to_records[raw_idx].append(rec)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records, expanded_names, raw_to_records


def raw_blocks(raw_indices, block_size):
    raw_indices = list(raw_indices)
    for i in range(0, len(raw_indices), block_size):
        yield raw_indices[i:i + block_size]


def auto_sizes(max_mem_bytes, avg_width, user_row_chunksize=0, user_raw_col_block=0):
    budget = max(int(max_mem_bytes * 0.45), 64 * 1024**2)
    if user_row_chunksize and user_row_chunksize > 0:
        row_chunksize = int(user_row_chunksize)
    else:
        row_chunksize = 4096
    if user_raw_col_block and user_raw_col_block > 0:
        raw_col_block = int(user_raw_col_block)
    else:
        bytes_per_raw_col_per_row = max(48, 16 * max(avg_width, 1.0))
        raw_col_block = max(1, int(budget * 0.6 / (row_chunksize * bytes_per_raw_col_per_row)))
        # raw_col_block = min(raw_col_block, 20000)
    return row_chunksize, raw_col_block


# -----------------------------
# CSV reading, bit extraction, TOG pass-through
# -----------------------------

# -----------------------------
# Fast bit-level CSV reader
# -----------------------------
#new add fast read zhangyu
def is_fast_bit_layout(layout):
    """Fast reader only supports already bit-level CSV: every expanded record width == 1."""
    return all(int(r.get("width", 1)) == 1 for r in layout)


def _parse_bit_tail_fast(tail, start_sig_col, ncols, expected_total_signal_cols=None):
    """
    Parse a contiguous block from CSV tail bytes.

    CSV data line format assumed:
        time,0,1,0,1,x,z,...

    After removing time and first comma, tail should be:
        0,1,0,1,x,z,...

    start_sig_col is 0-based among signal columns, not including time.
    ncols is current block width.

    Return:
        level_row: uint8 array, shape=(ncols,), value 0/1
        dirty_row: bool array, shape=(ncols,), True for x/z/invalid
    """
    tail = tail.rstrip(b"\r\n")

    # Fast path for single-character comma-separated values.
    # Signal column j is at byte offset 2*j in tail.
    start_byte = 2 * int(start_sig_col)
    end_byte = start_byte + 2 * int(ncols) - 1

    if end_byte <= len(tail):
        vals = np.frombuffer(tail, dtype=np.uint8)[start_byte:end_byte:2]

        if vals.size == ncols:
            level_row = (vals == ord("1")).astype(np.uint8)
            dirty_row = (vals != ord("0")) & (vals != ord("1"))
            return level_row, dirty_row

    # Slow fallback: robust split. This should rarely be used for your bit-level CSV.
    parts = tail.split(b",")
    if expected_total_signal_cols is not None and len(parts) != expected_total_signal_cols:
        raise ValueError(
            f"CSV data column count mismatch: got={len(parts)}, "
            f"expected={expected_total_signal_cols}"
        )

    block_parts = parts[start_sig_col:start_sig_col + ncols]
    if len(block_parts) != ncols:
        raise ValueError(
            f"CSV block slice mismatch: got={len(block_parts)}, expected={ncols}, "
            f"start_sig_col={start_sig_col}"
        )

    vals = np.empty(ncols, dtype=np.uint8)
    dirty = np.zeros(ncols, dtype=bool)

    for i, p in enumerate(block_parts):
        p = p.strip()
        if p == b"1":
            vals[i] = 1
        elif p == b"0":
            vals[i] = 0
        else:
            vals[i] = 0
            dirty[i] = True

    return vals, dirty


def iter_fast_bit_block_chunks(csv_path, raw_indices, row_chunksize, total_signal_cols):
    """
    Fast replacement for pandas read_csv + extract_bit_matrix, for bit-level CSV only.

    raw_indices are original CSV column indices including time column.
    Example:
      raw_idx=0 -> time
      raw_idx=1 -> first signal column

    This function assumes raw_indices is a contiguous block produced by raw_blocks().
    It yields:
      level_mat: uint8 matrix, shape=(rows_in_chunk, ncols)
      xz_flags: bool vector, shape=(ncols,), True if this column has x/z/invalid in this chunk
    """
    if not raw_indices:
        return

    # raw_indices include time as col 0, but signal tail excludes time.
    # So raw_idx=1 maps to signal col 0.
    start_sig_col = int(raw_indices[0]) - 1
    ncols = len(raw_indices)

    # Safety check: this fast reader only supports contiguous column blocks.
    expected = list(range(raw_indices[0], raw_indices[0] + ncols))
    if list(raw_indices) != expected:
        raise ValueError("fast bit reader requires contiguous raw_indices block")

    rows = []
    dirty_acc = np.zeros(ncols, dtype=bool)

    f, closer = open_csv_binary(csv_path)
    try:
        _header = f.readline()  # skip header

        for line_no, line in enumerate(f, 2):
            if not line.strip():
                continue

            comma = line.find(b",")
            if comma < 0:
                raise ValueError(f"{csv_path}:{line_no}: bad CSV line, no comma found")

            tail = line[comma + 1:]
            level_row, dirty_row = _parse_bit_tail_fast(
                tail=tail,
                start_sig_col=start_sig_col,
                ncols=ncols,
                expected_total_signal_cols=total_signal_cols,
            )

            rows.append(level_row)
            dirty_acc |= dirty_row

            if len(rows) >= row_chunksize:
                level_mat = np.vstack(rows).astype(np.uint8, copy=False)
                yield level_mat, dirty_acc.copy()
                rows.clear()
                dirty_acc[:] = False

        if rows:
            level_mat = np.vstack(rows).astype(np.uint8, copy=False)
            yield level_mat, dirty_acc.copy()
    finally:
        closer()


def _read_csv_content_as_text(csv_path):
    """Read entire CSV content as a string, supporting .csv and .csv.tgz files."""
    if _is_tgz(csv_path):
        proc = subprocess.run(
            ['tar', 'xzf', str(csv_path), '-O'],
            capture_output=True,
            check=True,
        )
        return proc.stdout.decode('utf-8')
    else:
        with open(str(csv_path), 'r', encoding='utf-8') as f:
            return f.read()


def iter_raw_block_chunks(csv_path, raw_indices, row_chunksize):
    """Legacy pandas reader with tgz support. For tgz files, content is read into memory first."""
    if _is_tgz(csv_path):
        content = _read_csv_content_as_text(csv_path)
        buf = io.StringIO(content)
        for df in pd.read_csv(
            buf,
            header=0,
            usecols=raw_indices,
            chunksize=row_chunksize,
            dtype=str,
            low_memory=False,
            engine="c",
        ):
            yield df
        buf.close()
        del content, buf
    else:
        for df in pd.read_csv(
            csv_path,
            header=0,
            usecols=raw_indices,
            chunksize=row_chunksize,
            dtype=str,
            low_memory=False,
            engine="c",
        ):
            yield df


def extract_bit_matrix(df, raw_to_records, raw_indices):
    """Return names, level uint8 dense matrix, xz flags per expanded column."""
    mats, names, xz_flags = [], [], []
    nrows = len(df)
    for raw_idx in raw_indices:
        recs = raw_to_records.get(raw_idx, [])
        if not recs:
            continue
        raw_name = recs[0]["raw_name"]
        if raw_name not in df.columns:
            raise KeyError(f"raw column not found after pandas read: {raw_name}")
        s = df[raw_name].fillna("0").astype(str).str.strip()
        arr = s.to_numpy(dtype=str)
        xz_row = s.str.contains(r"[xXzZ]", regex=True).to_numpy(dtype=bool)
        width = int(recs[0]["width"])
        if width == 1:
            v = arr
            col = np.zeros(nrows, dtype=np.uint8)
            col[v == "1"] = 1
            dirty = xz_row | ~((v == "0") | (v == "1"))
            mats.append(col[:, None])
            names.append(recs[0]["name"])
            xz_flags.append(bool(dirty.any()))
        else:
            bad_len = np.fromiter((len(x) != width for x in arr), dtype=bool, count=nrows)
            dirty_parent = xz_row | bad_len
            for rec in recs:
                bit_pos = int(rec["bit_pos"])
                vals = np.fromiter((x[bit_pos] if len(x) > bit_pos else "0" for x in arr), dtype="U1", count=nrows)
                col = np.zeros(nrows, dtype=np.uint8)
                col[vals == "1"] = 1
                dirty = dirty_parent | ~((vals == "0") | (vals == "1"))
                mats.append(col[:, None])
                names.append(rec["name"])
                xz_flags.append(bool(dirty.any()))
    if not mats:
        return [], np.zeros((nrows, 0), dtype=np.uint8), np.zeros(0, dtype=bool)
    return names, np.hstack(mats), np.array(xz_flags, dtype=bool)


def passthrough_tog(tog_mat):
    """Input data is already TOG; keep rows unchanged."""
    return tog_mat.astype(np.uint8, copy=False)


def level_to_tog_sparse(level_csr, row_chunk=2048):
    """Convert level sparse CSR matrix to TOG sparse CSR.
    TOG[t] = Level[t+1] XOR Level[t], for t = 0..n-2.
    TOG has n-1 rows; first level row is implicitly dropped.
    Processes in dense row-chunks to balance memory and speed.
    """
    n_rows, n_cols = level_csr.shape
    if n_rows <= 1:
        return sparse.csr_matrix((0, n_cols), dtype=np.uint8)

    parts = []
    for start in range(0, n_rows - 1, row_chunk):
        end = min(start + row_chunk + 1, n_rows)
        # Read enough rows to compute TOG for [start, end-1)
        chunk = level_csr[start:end].toarray().astype(np.uint8)
        # Compute XOR diff: TOG[t] = (chunk[t+1] != chunk[t])
        tog_dense = (chunk[1:] != chunk[:-1]).astype(np.uint8)
        parts.append(sparse.csr_matrix(tog_dense, dtype=np.uint8))
        del chunk, tog_dense

    out = sparse.vstack(parts, format='csr', dtype=np.uint8)
    del parts
    gc.collect()
    return out


# -----------------------------
# cache generation
# -----------------------------
def make_cache_path(cache_dir, csv_path, block_id, fmt, data_type="tog", row_shard=None):
    tag = "level" if data_type == "level" else "tog"
    base = str(Path(cache_dir) / f"{safe_stem(csv_path)}.block{block_id:05d}.{tag}.cache.{fmt}")
    if row_shard is not None:
        return f"{base}.row{int(row_shard):05d}.npz"
    return f"{base}.npz"


def _save_cache_matrix_zip64_safe(mat, cache_dir, csv_path, block_id, cache_format, data_type):
    """Save a sparse matrix as cache .npz, row-sharding if needed to avoid ZIP64.

    Returns (paths, n_rows_total, nnz_total).
      - paths: single str when under the ZIP entry limit, list of str otherwise.
      - n_rows_total: total rows (same as mat.shape[0]).
      - nnz_total: total nnz (same as mat.nnz).

    The caller stores paths into the manifest; load_cache_block_matrix handles
    both single-path (str) and sharded (list) transparently.
    """
    n_rows = int(mat.shape[0])
    n_cols = int(mat.shape[1])
    nnz_val = int(mat.nnz)

    full_npz = _total_npz_bytes(n_rows, n_cols, nnz_val, fmt=cache_format, dtype_bytes=1)

    if full_npz <= _ZIP_ENTRY_BYTES_LIMIT:
        single_path = make_cache_path(cache_dir, csv_path, block_id, cache_format, data_type=data_type)
        save_npz(single_path, mat, compressed=True)
        log(f"CACHE SAVE ({data_type}): {csv_path} block={block_id} -> {single_path}, shape={mat.shape}, nnz={mat.nnz}")
        return single_path, n_rows, nnz_val

    # Row-shard: each nnz contributes (dtype_bytes + idx_b) bytes to total .npz size.
    # Target total <= _ZIP_ENTRY_BYTES_LIMIT per shard.
    nnz_per_row = float(nnz_val) / max(1.0, float(n_rows))
    idx_b = _index_dtype_bytes(n_rows, n_cols, nnz_val)
    bytes_per_nnz = 1 + idx_b  # data (uint8) + indices
    max_rows_per_shard = max(1, int(float(_ZIP_ENTRY_BYTES_LIMIT) / max(1e-9, nnz_per_row * float(bytes_per_nnz))))
    max_rows_per_shard = min(max_rows_per_shard, n_rows)

    log(
        f"ZIP64 WARNING cache ({data_type}): {csv_path} block={block_id} "
        f"total_npz={full_npz / 1024 ** 3:.2f}GiB > "
        f"ZIP limit={_ZIP_ENTRY_BYTES_LIMIT / 1024 ** 3:.2f}GiB; "
        f"row-sharding with max_rows_per_shard={max_rows_per_shard}"
    )

    paths = []
    for si, r0 in enumerate(range(0, n_rows, max_rows_per_shard)):
        r1 = min(n_rows, r0 + max_rows_per_shard)
        shard = mat[r0:r1]
        shard_path = make_cache_path(cache_dir, csv_path, block_id, cache_format, data_type=data_type, row_shard=si)
        save_npz(shard_path, shard, compressed=True)
        paths.append(shard_path)
        log(
            f"CACHE SAVE ({data_type}) shard {si}: {csv_path} block={block_id} "
            f"rows={r0}:{r1} -> {shard_path}, shape={shard.shape}, nnz={shard.nnz}"
        )
        del shard

    return paths, n_rows, nnz_val


#newmodify zhangyu
#def build_tog_cache_once(csv_paths, raw_to_records, all_raw_indices, row_chunksize, raw_col_block, cache_dir, cache_format):
def build_tog_cache_once(
    csv_paths,
    raw_to_records,
    all_raw_indices,
    row_chunksize,
    raw_col_block,
    cache_dir,
    cache_format,
    use_fast_bit_reader=False,
    total_signal_cols=None,
    input_data_type="tog",
):
    """Read CSV only once and save after-xz-clean cache by raw-column block.

    When input_data_type='level', produces two caches:
      - level cache (all rows, X/Z cleaned)
      - TOG cache (XOR diff applied, n-1 rows, X/Z cleaned)

    Returns (manifest, xz_removed, [tog_manifest_if_level]).
    """
    mkdirp(cache_dir)
    manifest = OrderedDict()
    manifest["cache_format"] = cache_format
    manifest["input_data_type"] = input_data_type
    manifest["tog_diff_applied"] = False
    manifest["tog_note"] = "Input CSV is assumed to be preprocessed TOG data; no XOR diff is applied and rows are not dropped." if input_data_type == "tog" else "Input CSV is level data; saved as-is (all rows). TOG version is derived via XOR diff."
    manifest["blocks"] = []
    manifest["csv_paths"] = csv_paths
    xz_removed = []
    all_clean_names = []
    total_raw_blocks = 0

    # For level input, build a parallel TOG manifest
    tog_manifest = None
    if input_data_type == "level":
        tog_manifest = OrderedDict()
        tog_manifest["cache_format"] = cache_format
        tog_manifest["input_data_type"] = "tog"
        tog_manifest["tog_diff_applied"] = True
        tog_manifest["tog_note"] = "TOG = Level[t+1] XOR Level[t]; first level row dropped. Derived from level input CSV."
        tog_manifest["blocks"] = []
        tog_manifest["csv_paths"] = csv_paths
        log("INPUT DATA TYPE: level → will produce both LEVEL cache (all rows) and TOG cache (n-1 rows)")

    for block_id, block in enumerate(raw_blocks(all_raw_indices, raw_col_block), 1):
        total_raw_blocks += 1
        block_names = [r["name"] for raw in block for r in raw_to_records[raw]]
        name_to_local = {n: i for i, n in enumerate(block_names)}
        xz_flags_global = np.zeros(len(block_names), dtype=bool)
        per_file_chunk_mats = {csv_path: [] for csv_path in csv_paths}
        per_file_rows = {csv_path: 0 for csv_path in csv_paths}

        log(f"CACHE block {block_id}: raw_cols={len(block)}, expanded_cols={len(block_names)}")

        for csv_path in csv_paths:
            if input_data_type == "level":
                log(f"LEVEL INPUT {csv_path}: 读取level数据，保存level+TOG两份cache")
            else:
                log(f"TOG INPUT {csv_path}: 已假定输入为TOG数据，跳过level->tog diff，保留全部有效行")

            if use_fast_bit_reader:
                # Fast path: bypass pandas for already bit-level CSV.
                t_csv = time.time()
                chunk_i = 0

                for level_mat, xz_flags in iter_fast_bit_block_chunks(
                    csv_path=csv_path,
                    raw_indices=block,
                    row_chunksize=row_chunksize,
                    total_signal_cols=total_signal_cols,
                ):
                    chunk_i += 1

                    # In fast bit mode, names are exactly block_names.
                    for j, n in enumerate(block_names):
                        xz_flags_global[name_to_local[n]] |= bool(xz_flags[j])

                    tog_mat = passthrough_tog(level_mat)

                    if tog_mat.shape[0] > 0:
                        per_file_chunk_mats[csv_path].append(
                            sparse.csr_matrix(tog_mat, dtype=np.uint8)
                        )
                        per_file_rows[csv_path] += int(tog_mat.shape[0])

                    log(
                        f"FAST chunk done: csv={safe_stem(csv_path)}, "
                        f"block={block_id}, chunk={chunk_i}, "
                        f"input_tog_shape={level_mat.shape}, cached_tog_shape={tog_mat.shape}, "
                        f"seconds={time.time() - t_csv:.1f}"
                    )

                    del level_mat, tog_mat
                    gc.collect()

            else:
                # Original pandas path.
                for df in iter_raw_block_chunks(csv_path, block, row_chunksize):
                    names, level_mat, xz_flags = extract_bit_matrix(df, raw_to_records, block)
                    if not names:
                        continue

                    # names should match block_names order for this block.
                    for j, n in enumerate(names):
                        xz_flags_global[name_to_local[n]] |= bool(xz_flags[j])

                    tog_mat = passthrough_tog(level_mat)

                    if tog_mat.shape[0] > 0:
                        per_file_chunk_mats[csv_path].append(
                            sparse.csr_matrix(tog_mat, dtype=np.uint8)
                        )
                        per_file_rows[csv_path] += int(tog_mat.shape[0])

                    del df, level_mat, tog_mat

            gc.collect()
        clean_idx = np.flatnonzero(~xz_flags_global)
        clean_names = [block_names[i] for i in clean_idx]
        removed_names = [block_names[i] for i in np.flatnonzero(xz_flags_global)]
        xz_removed.extend(removed_names)
        all_clean_names.extend(clean_names)

        block_entry = OrderedDict()
        block_entry["block_id"] = block_id
        block_entry["raw_indices"] = block
        block_entry["names"] = clean_names
        block_entry["removed_xz_names"] = removed_names
        block_entry["files"] = OrderedDict()
        block_entry["n_rows"] = OrderedDict()

        # Parallel TOG block entry if level input
        tog_block_entry = None
        if tog_manifest is not None:
            tog_block_entry = OrderedDict()
            tog_block_entry["block_id"] = block_id
            tog_block_entry["raw_indices"] = list(block)
            tog_block_entry["names"] = list(clean_names)  # same X/Z cleaning
            tog_block_entry["removed_xz_names"] = list(removed_names)
            tog_block_entry["files"] = OrderedDict()
            tog_block_entry["n_rows"] = OrderedDict()

        for csv_path in csv_paths:
            if per_file_chunk_mats[csv_path]:
                mat = sparse.vstack(per_file_chunk_mats[csv_path], format="csr", dtype=np.uint8)
            else:
                mat = sparse.csr_matrix((0, len(block_names)), dtype=np.uint8)
            # after x/z clean
            mat = mat[:, clean_idx]

            # ------ Save LEVEL cache ------
            level_cache_type = "level" if input_data_type == "level" else "tog"
            if cache_format == "csc":
                mat_level = mat.tocsc()
            else:
                mat_level = mat.tocsr()
            level_paths, level_n_rows, level_nnz = _save_cache_matrix_zip64_safe(
                mat_level, cache_dir, csv_path, block_id, cache_format, data_type=level_cache_type,
            )
            block_entry["files"][csv_path] = level_paths
            block_entry["n_rows"][csv_path] = int(level_n_rows)
            if "nnz" not in block_entry:
                block_entry["nnz"] = OrderedDict()
            block_entry["nnz"][csv_path] = int(level_nnz)

            # ------ Save TOG cache if level input ------
            if tog_block_entry is not None:
                t_tog = time.time()
                if mat_level.shape[0] > 1:
                    # Convert level → TOG: XOR diff, keep n-1 rows (drop first row)
                    mat_tog = level_to_tog_sparse(mat_level.tocsr())
                else:
                    # Empty or single-row level data → empty TOG
                    mat_tog = sparse.csr_matrix((0, mat_level.shape[1]), dtype=np.uint8)
                if cache_format == "csc":
                    mat_tog = mat_tog.tocsc()
                else:
                    mat_tog = mat_tog.tocsr()
                tog_paths, tog_n_rows, tog_nnz = _save_cache_matrix_zip64_safe(
                    mat_tog, cache_dir, csv_path, block_id, cache_format, data_type="tog",
                )
                tog_block_entry["files"][csv_path] = tog_paths
                tog_block_entry["n_rows"][csv_path] = int(tog_n_rows)
                if "nnz" not in tog_block_entry:
                    tog_block_entry["nnz"] = OrderedDict()
                tog_block_entry["nnz"][csv_path] = int(tog_nnz)
                log(f"CACHE TOG convert: {csv_path} block={block_id} tog_seconds={time.time() - t_tog:.1f}")
                del mat_tog

            del mat_level, mat

        manifest["blocks"].append(block_entry)
        if tog_block_entry is not None:
            tog_manifest["blocks"].append(tog_block_entry)

        del per_file_chunk_mats
        gc.collect()

    manifest["clean_names_after_xz"] = all_clean_names
    manifest["xz_removed_count"] = len(xz_removed)
    manifest["num_blocks"] = total_raw_blocks
    if tog_manifest is not None:
        tog_manifest["clean_names_after_xz"] = all_clean_names
        tog_manifest["xz_removed_count"] = len(xz_removed)
        tog_manifest["num_blocks"] = total_raw_blocks
        return manifest, xz_removed, tog_manifest
    return manifest, xz_removed


def save_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def load_cache_block_matrix(block_entry, csv_path, fmt="csc"):
    file_ref = block_entry.get("files", {}).get(csv_path)
    if file_ref is None:
        # File not in this block (e.g., empty TOG for a level file with 0 rows)
        ncols = len(block_entry.get("names", []))
        if fmt == "csc":
            return sparse.csc_matrix((0, ncols), dtype=np.uint8)
        else:
            return sparse.csr_matrix((0, ncols), dtype=np.uint8)
    if isinstance(file_ref, list):
        # Row-sharded cache block: load and vstack
        mats = [load_npz(p).asformat(fmt) for p in file_ref]
        out = sparse.vstack(mats, format=fmt, dtype=np.uint8)
        for m in mats:
            del m
        return out
    mat = load_npz(file_ref)
    return mat.asformat(fmt)


# -----------------------------
# column stats from CSC cache
# -----------------------------
def hash_sparse_col(h, nrows, indices):
    h.update(np.int64(nrows).tobytes())
    h.update(np.int64(len(indices)).tobytes())
    if len(indices):
        h.update(np.asarray(indices, dtype=np.int64).tobytes())


def collect_stats_from_cache(manifest, stats_jsonl):
    if os.path.exists(stats_jsonl):
        os.remove(stats_jsonl)
    stats = []
    for block_entry in manifest["blocks"]:
        names = block_entry["names"]
        ncols = len(names)
        states = []
        for n in names:
            states.append({
                "name": n,
                "all_zero": True,
                "all_one": True,
                "rows": 0,
                "hasher": hashlib.blake2b(digest_size=16),
            })
        log(f"STATS from cache block {block_entry['block_id']}: cols={ncols}")
        for csv_path in manifest["csv_paths"]:
            mat = load_cache_block_matrix(block_entry, csv_path, fmt="csc")
            nrows = int(mat.shape[0])
            nnz_per_col = np.diff(mat.indptr)
            for j in range(ncols):
                st = states[j]
                st["all_zero"] = st["all_zero"] and bool(nnz_per_col[j] == 0)
                st["all_one"] = st["all_one"] and bool(nnz_per_col[j] == nrows)
                st["rows"] += nrows
                start, end = mat.indptr[j], mat.indptr[j + 1]
                hash_sparse_col(st["hasher"], nrows, mat.indices[start:end])
            del mat
            gc.collect()
        with open(stats_jsonl, "a", encoding="utf-8") as f:
            for st in states:
                out = {k: v for k, v in st.items() if k != "hasher"}
                out["digest"] = st["hasher"].hexdigest()
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats.append(out)
    return stats


def build_stage_columns(stats, keep_all_one_one_col=True):
    after_xz = [r["name"] for r in stats]
    zero_removed = []
    after_zero = []
    for r in stats:
        if r["all_zero"]:
            zero_removed.append(r["name"])
        else:
            after_zero.append(r["name"])

    after_one = []
    all_one_cols = []
    removed_one = []
    kept_one = None
    after_zero_set = set(after_zero)
    for r in stats:
        name = r["name"]
        if name not in after_zero_set:
            continue
        if r["all_one"]:
            all_one_cols.append(name)
            if keep_all_one_one_col:
                if kept_one is None:
                    kept_one = name
                    after_one.append(name)
                else:
                    removed_one.append(name)
            else:
                removed_one.append(name)
        else:
            after_one.append(name)
    return {
        "after_xz_clean": after_xz,
        "after_zero": after_zero,
        "zero_removed": zero_removed,
        "after_one": after_one,
        "all_one_cols": all_one_cols,
        "kept_all_one_col": kept_one,
        "removed_all_one_cols": removed_one,
    }


# -----------------------------
# cache index utilities
# -----------------------------
def build_name_index(manifest):
    name_to_block = {}
    for bidx, block in enumerate(manifest["blocks"]):
        for local_idx, name in enumerate(block["names"]):
            name_to_block[name] = (bidx, local_idx)
    return name_to_block


def names_by_block(names, name_to_block):
    grouped = defaultdict(list)
    for name in names:
        if name in name_to_block:
            bidx, local = name_to_block[name]
            grouped[bidx].append((name, local))
    return grouped


# -----------------------------
# exact / hash-based duplicate removal from cache
# -----------------------------
def exact_vectors_for_names(manifest, names, name_to_block=None, verbose=False):
    """Return exact sparse byte representation of selected names across all CSVs.

    Bucket strategy helper. It may repeatedly load cache blocks if many buckets
    are verified, so it is mainly kept as a low-memory fallback.
    """
    if name_to_block is None:
        name_to_block = build_name_index(manifest)
    bufs = {n: bytearray() for n in names}
    grouped = names_by_block(names, name_to_block)

    t_func = time.time()
    for bidx, items in grouped.items():
        block = manifest["blocks"][bidx]
        t_block = time.time()
        if verbose:
            log(f"DUP exact bucket-load: block={block['block_id']}, selected_cols={len(items)}, csv_files={len(manifest['csv_paths'])}")
        for csv_i, csv_path in enumerate(manifest["csv_paths"], 1):
            t_load = time.time()
            mat = load_cache_block_matrix(block, csv_path, fmt="csc")
            load_s = time.time() - t_load
            nrows = int(mat.shape[0])
            t_proc = time.time()
            for name, local in items:
                start, end = mat.indptr[local], mat.indptr[local + 1]
                idx = mat.indices[start:end]
                data = mat.data[start:end]
                bufs[name].extend(np.int64(nrows).tobytes())
                bufs[name].extend(np.int64(len(idx)).tobytes())
                if len(idx):
                    bufs[name].extend(np.asarray(idx, dtype=np.int64).tobytes())
                    # Cache data should be binary 1s, but include data to make exact
                    # verification robust if a non-binary sparse matrix is ever used.
                    bufs[name].extend(np.asarray(data).tobytes())
            proc_s = time.time() - t_proc
            if verbose:
                log(
                    f"DUP exact bucket-load csv {csv_i}/{len(manifest['csv_paths'])}: "
                    f"block={block['block_id']}, rows={nrows}, nnz={mat.nnz}, "
                    f"load_s={load_s:.2f}, proc_s={proc_s:.2f}"
                )
            del mat
            gc.collect()
        if verbose:
            log(f"DUP exact bucket-load block done: block={block['block_id']}, seconds={time.time() - t_block:.1f}")
    if verbose:
        total_bytes = sum(len(v) for v in bufs.values())
        log(f"DUP exact bucket-load done: names={len(names)}, signature_bytes={total_bytes/1024**2:.1f} MiB, seconds={time.time() - t_func:.1f}")
    return {n: bytes(bufs[n]) for n in names}


def _prepare_duplicate_buckets(stats, candidate_cols):
    cand_set = set(candidate_cols)
    digest_to_names = defaultdict(list)
    for r in stats:
        n = r["name"]
        if n in cand_set:
            digest_to_names[r["digest"]].append(n)
    return digest_to_names


def _log_duplicate_bucket_summary(digest_to_names, top_k=10):
    sizes = [len(v) for v in digest_to_names.values()]
    total_buckets = len(sizes)
    multi_sizes = [x for x in sizes if x > 1]
    multi_buckets = len(multi_sizes)
    multi_cols = sum(multi_sizes)
    max_bucket = max(sizes) if sizes else 0
    log(
        f"DUP bucket summary: buckets={total_buckets}, singleton_buckets={total_buckets - multi_buckets}, "
        f"multi_buckets={multi_buckets}, multi_bucket_cols={multi_cols}, max_bucket={max_bucket}"
    )
    if multi_sizes:
        top = sorted(multi_sizes, reverse=True)[:top_k]
        log(f"DUP bucket summary: top_{min(top_k, len(top))}_bucket_sizes={top}")


def remove_duplicates_by_hash(stats, candidate_cols, progress_interval=200000):
    """Fast duplicate removal using the global sparse digest as the identity.

    The digest is computed from all CSVs jointly and includes each CSV's row count
    plus each column's non-zero row indices. Therefore two columns with the same
    digest are overwhelmingly likely to be exactly identical across all CSV files.
    """
    cand_set = set(candidate_cols)
    order = {n: i for i, n in enumerate(candidate_cols)}
    digest_to_keep = {}
    keep = []
    dup_map = OrderedDict()
    duplicate_groups = defaultdict(int)
    processed = 0
    t0 = time.time()

    log(f"DUP hash-only start: candidate_cols={len(candidate_cols)}, progress_interval={progress_interval}")
    for r in stats:
        n = r["name"]
        if n not in cand_set:
            continue
        d = r["digest"]
        if d in digest_to_keep:
            dup_map[n] = digest_to_keep[d]
            duplicate_groups[d] += 1
        else:
            digest_to_keep[d] = n
            keep.append(n)
        processed += 1
        if progress_interval and processed % progress_interval == 0:
            log(
                f"DUP hash-only progress: processed={processed}/{len(candidate_cols)}, "
                f"kept={len(keep)}, removed={len(dup_map)}, seconds={time.time() - t0:.1f}"
            )

    keep = sorted(keep, key=lambda x: order[x])
    large_buckets = [[d, c + 1] for d, c in duplicate_groups.items() if c + 1 > 1]
    large_buckets = sorted(large_buckets, key=lambda x: x[1], reverse=True)[:1000]
    log(
        f"DUP hash-only done: processed={processed}, kept={len(keep)}, "
        f"removed={len(dup_map)}, groups={len(duplicate_groups)}, seconds={time.time() - t0:.1f}"
    )
    return keep, dup_map, large_buckets


def remove_exact_duplicates_onepass(manifest, stats, candidate_cols, exact_group_limit=20000,
                                    progress_interval=200000, verbose=False):
    """Sparse exact duplicate removal using a one-pass cache scan.

    This follows the same idea as the original dense script:
      1. hash columns first;
      2. only columns in the same hash bucket require exact verification;
      3. exact verification compares the full column vector across all CSV files.

    Difference from the dense script:
      - it never builds a dense vstack matrix;
      - it uses CSC column non-zero row indices as the exact representation;
      - all non-singleton buckets are verified by scanning each cache block once.

    This is usually much faster than verifying each bucket separately, but it may
    use more memory because signatures for all columns in non-singleton buckets
    are kept until comparison.
    """
    t_all = time.time()
    order = {n: i for i, n in enumerate(candidate_cols)}
    digest_to_names = _prepare_duplicate_buckets(stats, candidate_cols)
    _log_duplicate_bucket_summary(digest_to_names)

    keep = []
    dup_map = OrderedDict()
    large_buckets = []

    multi_names = []
    for digest, names in digest_to_names.items():
        if len(names) == 1:
            keep.append(names[0])
        else:
            names = sorted(names, key=lambda x: order[x])
            digest_to_names[digest] = names
            multi_names.extend(names)
            if len(names) > exact_group_limit:
                large_buckets.append([digest, len(names)])
                log(f"WARNING: large duplicate hash bucket: digest={digest}, cols={len(names)}")

    if not multi_names:
        keep = sorted(keep, key=lambda x: order[x])
        log(f"DUP exact one-pass done: no multi-column buckets, seconds={time.time() - t_all:.1f}")
        return keep, dup_map, large_buckets

    name_to_block = build_name_index(manifest)
    grouped = names_by_block(multi_names, name_to_block)
    bufs = {n: bytearray() for n in multi_names}
    multi_set = set(multi_names)

    log(
        f"DUP exact one-pass start: candidate_cols={len(candidate_cols)}, "
        f"verify_cols={len(multi_names)}, cache_blocks_with_verify_cols={len(grouped)}, "
        f"csv_files={len(manifest['csv_paths'])}, exact_group_limit={exact_group_limit}"
    )

    processed_blocks = 0
    total_load_s = 0.0
    total_proc_s = 0.0
    total_nnz_seen = 0
    last_progress_cols = 0

    for bidx in sorted(grouped.keys()):
        items = grouped[bidx]
        block = manifest["blocks"][bidx]
        processed_blocks += 1
        t_block = time.time()
        block_load_s = 0.0
        block_proc_s = 0.0
        block_nnz = 0
        log(
            f"DUP exact one-pass block start: {processed_blocks}/{len(grouped)}, "
            f"block_id={block['block_id']}, verify_cols_in_block={len(items)}"
        )

        for csv_i, csv_path in enumerate(manifest["csv_paths"], 1):
            t_load = time.time()
            mat = load_cache_block_matrix(block, csv_path, fmt="csc")
            load_s = time.time() - t_load
            t_proc = time.time()
            nrows = int(mat.shape[0])
            for name, local in items:
                start, end = mat.indptr[local], mat.indptr[local + 1]
                idx = mat.indices[start:end]
                data = mat.data[start:end]
                bufs[name].extend(np.int64(nrows).tobytes())
                bufs[name].extend(np.int64(len(idx)).tobytes())
                if len(idx):
                    bufs[name].extend(np.asarray(idx, dtype=np.int64).tobytes())
                    bufs[name].extend(np.asarray(data).tobytes())
            proc_s = time.time() - t_proc
            block_load_s += load_s
            block_proc_s += proc_s
            block_nnz += int(mat.nnz)
            if verbose:
                log(
                    f"DUP exact one-pass csv progress: block_id={block['block_id']}, "
                    f"csv={csv_i}/{len(manifest['csv_paths'])}, rows={nrows}, nnz={mat.nnz}, "
                    f"load_s={load_s:.2f}, proc_s={proc_s:.2f}"
                )
            del mat
            gc.collect()

        total_load_s += block_load_s
        total_proc_s += block_proc_s
        total_nnz_seen += block_nnz
        sig_bytes = sum(len(bufs[n]) for n, _local in items)
        total_sig_bytes = sum(len(v) for v in bufs.values())
        log(
            f"DUP exact one-pass block done: block_id={block['block_id']}, "
            f"block_nnz={block_nnz}, block_sig_bytes={sig_bytes/1024**2:.1f} MiB, "
            f"total_sig_bytes={total_sig_bytes/1024**2:.1f} MiB, "
            f"load_s={block_load_s:.1f}, proc_s={block_proc_s:.1f}, "
            f"block_s={time.time() - t_block:.1f}, total_s={time.time() - t_all:.1f}"
        )

        if progress_interval:
            built_cols = sum(1 for n in multi_names if len(bufs[n]) > 0)
            if built_cols - last_progress_cols >= progress_interval:
                log(
                    f"DUP exact one-pass total progress: signature_built_cols≈{built_cols}/{len(multi_names)}, "
                    f"blocks={processed_blocks}/{len(grouped)}, total_load_s={total_load_s:.1f}, "
                    f"total_proc_s={total_proc_s:.1f}, total_nnz_seen={total_nnz_seen}, "
                    f"elapsed_s={time.time() - t_all:.1f}"
                )
                last_progress_cols = built_cols

    log(
        f"DUP exact one-pass compare start: multi_buckets={sum(1 for v in digest_to_names.values() if len(v) > 1)}, "
        f"verify_cols={len(multi_names)}, signature_bytes={sum(len(v) for v in bufs.values())/1024**2:.1f} MiB"
    )
    t_cmp = time.time()
    compared_cols = 0
    for bucket_i, (digest, names) in enumerate(digest_to_names.items(), 1):
        if len(names) == 1:
            continue
        exact_groups = OrderedDict()
        for n in names:
            v = bytes(bufs[n])
            if v in exact_groups:
                dup_map[n] = exact_groups[v]
            else:
                exact_groups[v] = n
                # Singleton bucket columns were already kept; multi bucket keepers are added here.
                if n in multi_set:
                    keep.append(n)
            compared_cols += 1
        if progress_interval and compared_cols % progress_interval == 0:
            log(
                f"DUP exact one-pass compare progress: compared_cols={compared_cols}/{len(multi_names)}, "
                f"kept={len(keep)}, removed={len(dup_map)}, compare_s={time.time() - t_cmp:.1f}"
            )

    keep = sorted(set(keep), key=lambda x: order[x])
    log(
        f"DUP exact one-pass done: processed={len(candidate_cols)}, kept={len(keep)}, "
        f"removed={len(dup_map)}, load_s={total_load_s:.1f}, proc_s={total_proc_s:.1f}, "
        f"compare_s={time.time() - t_cmp:.1f}, total_s={time.time() - t_all:.1f}"
    )
    return keep, dup_map, large_buckets


def remove_exact_duplicates_bucket(manifest, stats, candidate_cols, exact_group_limit=20000,
                                   progress_interval=200000, verbose=False):
    """Low-memory sparse exact duplicate removal, verifying buckets one by one.

    This is safer when non-singleton buckets are huge and one-pass signatures do
    not fit in memory, but it can be slower because cache blocks may be loaded
    repeatedly for different buckets.
    """
    digest_to_names = _prepare_duplicate_buckets(stats, candidate_cols)
    _log_duplicate_bucket_summary(digest_to_names)
    order = {n: i for i, n in enumerate(candidate_cols)}
    keep = []
    dup_map = OrderedDict()
    large_buckets = []
    name_to_block = build_name_index(manifest)

    total_buckets = len(digest_to_names)
    log(
        f"DUP exact bucket start: candidate_cols={len(candidate_cols)}, buckets={total_buckets}, "
        f"exact_group_limit={exact_group_limit}, progress_interval={progress_interval}"
    )

    t0 = time.time()
    processed_cols = 0
    next_progress = progress_interval
    for bucket_i, (digest, names) in enumerate(digest_to_names.items(), 1):
        names = sorted(names, key=lambda x: order[x])
        if len(names) == 1:
            keep.append(names[0])
            processed_cols += 1
        else:
            if len(names) > exact_group_limit:
                large_buckets.append([digest, len(names)])
                log(f"WARNING: large duplicate hash bucket: digest={digest}, cols={len(names)}")

            log(f"DUP exact bucket verify start: bucket={bucket_i}/{total_buckets}, cols={len(names)}")
            exact_groups = OrderedDict()
            for start_i in range(0, len(names), exact_group_limit):
                batch = names[start_i:start_i + exact_group_limit]
                t_batch = time.time()
                vecs = exact_vectors_for_names(manifest, batch, name_to_block=name_to_block, verbose=verbose)
                load_compare_s = time.time() - t_batch
                for n in batch:
                    v = vecs[n]
                    if v in exact_groups:
                        dup_map[n] = exact_groups[v]
                    else:
                        exact_groups[v] = n
                        keep.append(n)
                processed_cols += len(batch)
                log(
                    f"DUP exact bucket batch done: bucket={bucket_i}/{total_buckets}, "
                    f"batch={start_i // exact_group_limit + 1}, batch_cols={len(batch)}, "
                    f"batch_s={load_compare_s:.1f}, processed_cols={processed_cols}/{len(candidate_cols)}, "
                    f"kept={len(keep)}, removed={len(dup_map)}, total_s={time.time() - t0:.1f}"
                )
                del vecs
                gc.collect()

        if progress_interval and processed_cols >= next_progress:
            log(
                f"DUP exact bucket total progress: processed_cols={processed_cols}/{len(candidate_cols)}, "
                f"bucket={bucket_i}/{total_buckets}, kept={len(keep)}, removed={len(dup_map)}, "
                f"seconds={time.time() - t0:.1f}"
            )
            while processed_cols >= next_progress:
                next_progress += progress_interval

    keep = sorted(keep, key=lambda x: order[x])
    log(
        f"DUP exact bucket done: processed={processed_cols}, kept={len(keep)}, "
        f"removed={len(dup_map)}, seconds={time.time() - t0:.1f}"
    )
    return keep, dup_map, large_buckets


def remove_exact_duplicates(manifest, stats, candidate_cols, exact_group_limit=20000,
                            progress_interval=200000, strategy="one-pass", verbose=False):
    """Dispatch exact duplicate removal strategy."""
    if strategy == "one-pass":
        return remove_exact_duplicates_onepass(
            manifest, stats, candidate_cols,
            exact_group_limit=exact_group_limit,
            progress_interval=progress_interval,
            verbose=verbose,
        )
    if strategy == "bucket":
        return remove_exact_duplicates_bucket(
            manifest, stats, candidate_cols,
            exact_group_limit=exact_group_limit,
            progress_interval=progress_interval,
            verbose=verbose,
        )
    raise ValueError(f"unknown duplicate exact strategy: {strategy}")


# -----------------------------
# similarity reduction from CSC cache
# -----------------------------
def build_minhash_params(num_perm, seed):
    rng = np.random.default_rng(seed)
    prime = np.uint64(4294967311)
    a = rng.integers(1, prime - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, prime - 1, size=num_perm, dtype=np.uint64)
    return a, b, prime


def minhash_indices(indices, global_row_offset, a, b, prime):
    if len(indices) == 0:
        return np.full(len(a), np.iinfo(np.uint64).max, dtype=np.uint64)
    rows = np.asarray(indices, dtype=np.uint64) + np.uint64(global_row_offset)
    hashes = (rows[:, None] * a[None, :] + b[None, :]) % prime
    return hashes.min(axis=0)


def lsh_candidates_from_signatures(names, sigs, bands, max_candidate_pairs):
    if sigs.shape[1] % bands != 0:
        raise ValueError("num_perm must be divisible by bands")
    rows_per_band = sigs.shape[1] // bands
    buckets = defaultdict(list)
    for i, n in enumerate(names):
        s = sigs[i]
        for b in range(bands):
            part = s[b * rows_per_band:(b + 1) * rows_per_band]
            key = (b, hashlib.blake2b(part.tobytes(), digest_size=8).digest())
            buckets[key].append(n)
    pairs = set()
    overflow = False
    for group in buckets.values():
        if len(group) < 2:
            continue
        group = sorted(group)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.add((group[i], group[j]))
                if len(pairs) >= max_candidate_pairs:
                    overflow = True
                    return pairs, overflow
    return pairs, overflow


def build_nz_sets_for_candidate_names(manifest, candidate_names):
    name_to_block = build_name_index(manifest)
    grouped = names_by_block(candidate_names, name_to_block)
    nz = {n: [] for n in candidate_names}
    # Need global row offset across CSVs. Same for all blocks.
    csv_offsets = OrderedDict()
    cur = 0
    # Use first non-empty block for row counts.
    for csv_path in manifest["csv_paths"]:
        nrows = None
        for b in manifest["blocks"]:
            if csv_path in b["n_rows"]:
                nrows = int(b["n_rows"][csv_path])
                break
        if nrows is None:
            nrows = 0
        csv_offsets[csv_path] = cur
        cur += nrows

    for bidx, items in grouped.items():
        block = manifest["blocks"][bidx]
        for csv_path in manifest["csv_paths"]:
            mat = load_cache_block_matrix(block, csv_path, fmt="csc")
            base = csv_offsets[csv_path]
            for name, local in items:
                start, end = mat.indptr[local], mat.indptr[local + 1]
                idx = mat.indices[start:end]
                if len(idx):
                    nz[name].append(idx.astype(np.int64) + base)
            del mat
            gc.collect()
    return {n: np.concatenate(parts) if parts else np.array([], dtype=np.int64) for n, parts in nz.items()}


def jaccard_sorted(a, b):
    i = j = inter = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] == b[j]:
            inter += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    union = la + lb - inter
    return 1.0 if union == 0 else inter / union


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def similarity_reduce_once(manifest, cols, threshold, num_perm, bands, seed, max_candidate_pairs):
    a, b, prime = build_minhash_params(num_perm, seed)
    col_set = set(cols)
    name_to_block = build_name_index(manifest)
    sig_by_name = {}

    # Prepare CSV global row offsets.
    csv_offsets = OrderedDict()
    cur = 0
    for csv_path in manifest["csv_paths"]:
        nrows = 0
        for block in manifest["blocks"]:
            if csv_path in block["n_rows"]:
                nrows = int(block["n_rows"][csv_path])
                break
        csv_offsets[csv_path] = cur
        cur += nrows

    for bidx, block in enumerate(manifest["blocks"]):
        items = [(n, local) for n, (bb, local) in name_to_block.items() if bb == bidx and n in col_set]
        if not items:
            continue
        local_sig = {n: np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64) for n, _ in items}
        for csv_path in manifest["csv_paths"]:
            mat = load_cache_block_matrix(block, csv_path, fmt="csc")
            base = csv_offsets[csv_path]
            for n, local in items:
                start, end = mat.indptr[local], mat.indptr[local + 1]
                sig = minhash_indices(mat.indices[start:end], base, a, b, prime)
                local_sig[n] = np.minimum(local_sig[n], sig)
            del mat
            gc.collect()
        sig_by_name.update(local_sig)
        log(f"SIM threshold={threshold}: signature block {block['block_id']} done, cols={len(items)}")

    names_all = [n for n in cols if n in sig_by_name]
    sigs = np.vstack([sig_by_name[n] for n in names_all]) if names_all else np.zeros((0, num_perm), dtype=np.uint64)
    pairs, overflow = lsh_candidates_from_signatures(names_all, sigs, bands, max_candidate_pairs)
    log(f"SIM threshold={threshold}: candidate_pairs={len(pairs)}, overflow={overflow}")

    candidate_names = sorted(set(x for p in pairs for x in p), key=lambda x: cols.index(x))
    nz_sets = build_nz_sets_for_candidate_names(manifest, candidate_names) if candidate_names else {}

    uf = UnionFind(cols)
    merged = 0
    for c1, c2 in pairs:
        if jaccard_sorted(nz_sets[c1], nz_sets[c2]) >= threshold:
            uf.union(c1, c2)
            merged += 1

    order = {c: i for i, c in enumerate(cols)}
    groups = defaultdict(list)
    for c in cols:
        groups[uf.find(c)].append(c)
    keep, drop = [], OrderedDict()
    for members in groups.values():
        members = sorted(members, key=lambda x: order[x])
        k = members[0]
        keep.append(k)
        for m in members[1:]:
            drop[m] = k
    keep = sorted(keep, key=lambda x: order[x])
    return keep, drop, {"candidate_pairs": len(pairs), "overflow": overflow, "merged_pairs": merged}


def progressive_similarity_reduce(manifest, cols, target_cols, thresholds, num_perm, bands, seed, max_candidate_pairs):
    current = list(cols)
    total_drop = OrderedDict()
    reports = []
    for th in thresholds:
        if len(current) <= target_cols:
            break
        t0 = time.time()
        keep, drop, rep = similarity_reduce_once(
            manifest, current, th, num_perm, bands, seed, max_candidate_pairs
        )
        report = {
            "threshold": th,
            "before": len(current),
            "after": len(keep),
            "removed": len(current) - len(keep),
            "seconds": time.time() - t0,
            **rep,
        }
        reports.append(report)
        for d, k in drop.items():
            while k in total_drop:
                k = total_drop[k]
            total_drop[d] = k
        current = keep
        log(f"SIM threshold={th}: before={report['before']}, after={report['after']}, removed={report['removed']}")
    return current, total_drop, reports


# -----------------------------
# stage/final snapshot saving with memory check
# -----------------------------
def stage_namefile_path(output_dir, stage_name):
    return str(Path(output_dir) / f"{stage_name}.namefile.txt")


def _index_dtype_bytes(n_rows, n_cols, nnz):
    """Approximate scipy sparse index dtype size."""
    max_val = max(int(n_rows), int(n_cols), int(nnz), 0)
    return 8 if max_val > np.iinfo(np.int32).max else 4


# Standard ZIP format cannot handle individual entries >= 4 GiB.
# NumPy < 2.0 does not support ZIP64 extensions in .npz files.
# We use a conservative threshold of 3.8 GiB to leave safety margin.
_ZIP_ENTRY_BYTES_LIMIT = int(3.8 * 1024 ** 3)


def _total_npz_bytes(n_rows, n_cols, nnz, fmt="csr", dtype_bytes=1):
    """Estimate total .npz file size for a sparse matrix.

    The .npz is a ZIP containing several .npy entries:
      - data.npy:    nnz * dtype_bytes
      - indices.npy: nnz * idx_b
      - indptr.npy:  indptr_len * idx_b
      - shape.npy / format.npy: ~500 bytes (headers + data)
      - NPY headers + ZIP local/central directory overhead: ~2000 bytes

    The TOTAL must stay under the ZIP 4 GiB limit — not just the largest
    entry, because scipy's save_npz / load_npz checks the overall file size.
    """
    idx_b = _index_dtype_bytes(n_rows, n_cols, nnz)
    if fmt == "csc":
        indptr_len = n_cols + 1
    else:
        indptr_len = n_rows + 1

    total = (
        int(nnz) * (int(dtype_bytes) + int(idx_b))   # data + indices
        + int(indptr_len) * int(idx_b)                # indptr
        + 2500                                         # shape/format/headers/ZIP overhead
    )
    return total


def estimate_sparse_matrix_bytes(n_rows, n_cols, nnz, fmt="csr", dtype_bytes=1, peak_factor=2.5):
    """Estimate peak memory needed to assemble and save one sparse matrix.

    This is intentionally conservative. It covers final sparse arrays plus
    temporary matrices during hstack/asformat/save_npz compression. It is an
    estimate, not a hard upper bound.
    """
    idx_b = _index_dtype_bytes(n_rows, n_cols, nnz)
    if fmt == "csc":
        indptr_len = n_cols + 1
    else:
        indptr_len = n_rows + 1
    base = int(nnz) * (dtype_bytes + idx_b) + int(indptr_len) * idx_b
    return int(base * peak_factor + 64 * 1024 * 1024)


def estimate_selected_output_stats(manifest, selected_names, csv_path, output_format="csr"):
    """Estimate n_rows, n_cols and nnz for selected columns of one CSV.

    The function scans cache blocks one at a time, so it does not assemble the
    full output matrix during estimation.
    """
    selected_set = set(selected_names)
    n_rows = 0
    n_cols = int(len(selected_names))
    nnz = 0
    got_rows = False

    for block in manifest["blocks"]:
        local_idx = [i for i, n in enumerate(block["names"]) if n in selected_set]
        if not got_rows and csv_path in block.get("n_rows", {}):
            n_rows = int(block["n_rows"][csv_path])
            got_rows = True
        if not local_idx:
            continue
        mat = load_cache_block_matrix(block, csv_path, fmt="csc")
        nnz_per_col = np.diff(mat.indptr)
        nnz += int(nnz_per_col[local_idx].sum())
        del mat
        gc.collect()

    est_bytes = estimate_sparse_matrix_bytes(n_rows, n_cols, nnz, fmt=output_format, dtype_bytes=1)
    return {"n_rows": int(n_rows), "n_cols": int(n_cols), "nnz": int(nnz), "estimated_peak_bytes": int(est_bytes)}


def _selected_local_pairs_for_block(block, selected_set, name_order):
    local_pairs = [(i, n) for i, n in enumerate(block["names"]) if n in selected_set]
    return sorted(local_pairs, key=lambda p: name_order[p[1]])


def assemble_selected_matrix_for_csv(manifest, selected_names, csv_path, output_format="csr", row_slice=None):
    """Assemble selected columns for one CSV.

    Columns are not split in the returned matrix. Internally the cache is stored
    by column blocks, so this function hstacks cache blocks only when the memory
    check says it is safe, or for a bounded row slice.
    """
    selected_set = set(selected_names)
    name_order = {n: i for i, n in enumerate(selected_names)}
    parts = []
    for block in manifest["blocks"]:
        local_pairs = _selected_local_pairs_for_block(block, selected_set, name_order)
        if not local_pairs:
            continue
        local_idx = [p[0] for p in local_pairs]
        mat = load_cache_block_matrix(block, csv_path, fmt="csc")
        if row_slice is None:
            sub = mat[:, local_idx]
        else:
            sub = mat[row_slice, :][:, local_idx]
        parts.append(sub.asformat(output_format))
        del mat, sub
        gc.collect()
    if not parts:
        n_rows = 0
        for block in manifest["blocks"]:
            if csv_path in block.get("n_rows", {}):
                n_rows = int(block["n_rows"][csv_path])
                break
        if row_slice is not None:
            start, stop = row_slice.start or 0, row_slice.stop if row_slice.stop is not None else n_rows
            n_rows = max(0, stop - start)
        return sparse.csr_matrix((n_rows, len(selected_names)), dtype=np.uint8).asformat(output_format)
    out = sparse.hstack(parts, format=output_format, dtype=np.uint8)
    del parts
    gc.collect()
    return out


def auto_row_block_rows(n_rows, full_estimated_bytes, memory_limit_bytes,
                        user_row_block_rows=0, total_nnz=0, zip_entry_limit=None):
    if user_row_block_rows and user_row_block_rows > 0:
        return max(1, int(user_row_block_rows))
    if n_rows <= 0:
        return 1

    rows = int(n_rows)

    # Constraint 1: peak memory per shard
    if full_estimated_bytes > 0 and memory_limit_bytes > 0:
        ratio = max(1e-9, (memory_limit_bytes * 0.70) / float(full_estimated_bytes))
        rows = int(max(1, min(n_rows, n_rows * ratio)))

    # Constraint 2: ZIP total file size per shard (avoid ZIP64 requirement)
    if total_nnz > 0 and zip_entry_limit and zip_entry_limit > 0:
        idx_b = _index_dtype_bytes(rows, 1, total_nnz)
        nnz_per_row = float(total_nnz) / float(n_rows)
        # Each nnz contributes (dtype_bytes + idx_b) bytes to total .npz size
        bytes_per_nnz = 1 + idx_b
        if nnz_per_row > 0:
            max_rows_for_zip = max(1, int(float(zip_entry_limit) / (nnz_per_row * float(bytes_per_nnz))))
            rows = min(rows, max_rows_for_zip)

    rows = max(1, min(rows, int(n_rows)))
    return rows


def save_stage_snapshot_memory_checked(
    manifest,
    selected_names,
    output_dir,
    stage_name,
    output_format="csr",
    max_mem_bytes=None,
    snapshot_mem_frac=0.60,
    row_block_rows=0,
    force_row_shard=False,
):
    """Save stage matrices without column splitting.

    Preferred output is one complete NPZ per input CSV:
        <csv_stem>.<stage>.tog.<format>.npz

    Before assembling the complete matrix, the function estimates peak memory
    AND checks the ZIP64 entry limit (single .npy array within the NPZ must
    be < 4 GiB for NumPy < 2.0 / SciPy < 1.12 compatibility).  If either
    check fails it automatically falls back to row-sharded output:
        <csv_stem>.<stage>.row00000.tog.<format>.npz
        <csv_stem>.<stage>.row00001.tog.<format>.npz

    In row-sharded mode, every shard keeps the full selected column set and
    only splits rows. Columns are never split in final/stage snapshots.
    """
    mkdirp(output_dir)
    namefile = stage_namefile_path(output_dir, stage_name)
    write_lines(namefile, selected_names)

    if max_mem_bytes is None:
        max_mem_bytes = parse_mem_to_bytes("28GB")
    memory_limit = int(max_mem_bytes * float(snapshot_mem_frac))

    snapshot = OrderedDict()
    snapshot["stage"] = stage_name
    snapshot["namefile"] = namefile
    snapshot["cols"] = int(len(selected_names))
    snapshot["format"] = output_format
    snapshot["column_sharded"] = False
    snapshot["memory_limit_bytes"] = int(memory_limit)
    snapshot["files"] = OrderedDict()

    for csv_path in manifest["csv_paths"]:
        est = estimate_selected_output_stats(manifest, selected_names, csv_path, output_format=output_format)

        # ZIP64 safety: check whether the TOTAL .npz file would exceed the
        # 4 GiB ZIP limit.  If so, force row-sharding even when peak memory
        # is within budget.
        zip64_total_npz = _total_npz_bytes(
            est["n_rows"], est["n_cols"], est["nnz"],
            fmt=output_format, dtype_bytes=1,
        )
        zip64_safe = zip64_total_npz <= _ZIP_ENTRY_BYTES_LIMIT
        est["zip64_total_npz_bytes"] = int(zip64_total_npz)
        est["zip64_safe"] = bool(zip64_safe)

        can_save_single = (
            (not force_row_shard)
            and (est["estimated_peak_bytes"] <= memory_limit)
            and zip64_safe
        )
        csv_record = OrderedDict()
        csv_record["estimated"] = est
        csv_record["single_file"] = bool(can_save_single)
        csv_record["row_sharded"] = not bool(can_save_single)

        if not zip64_safe and (not force_row_shard) and (est["estimated_peak_bytes"] <= memory_limit):
            log(
                f"ZIP64 WARNING {stage_name}: {csv_path} total_npz={zip64_total_npz/1024**3:.2f}GiB "
                f"> ZIP limit={_ZIP_ENTRY_BYTES_LIMIT/1024**3:.2f}GiB; "
                f"forcing row-shard to avoid 'ZIP64 is not implemented' error"
            )

        if can_save_single:
            log(
                f"SAVE {stage_name}: {csv_path} estimated_peak={est['estimated_peak_bytes']/1024**3:.2f}GB "
                f"<= limit={memory_limit/1024**3:.2f}GB, zip64_safe=True; saving single {output_format.upper()} NPZ"
            )
            mat = assemble_selected_matrix_for_csv(manifest, selected_names, csv_path, output_format=output_format)
            out = str(Path(output_dir) / f"{safe_stem(csv_path)}.{stage_name}.tog.{output_format}.npz")
            save_npz(out, mat, compressed=True)
            csv_record["path"] = out
            csv_record["shape"] = [int(mat.shape[0]), int(mat.shape[1])]
            csv_record["nnz"] = int(mat.nnz)
            log(f"SAVE {stage_name}: {csv_path} -> {out}, shape={mat.shape}, nnz={mat.nnz}")
            del mat
            gc.collect()
        else:
            n_rows = int(est["n_rows"])
            total_nnz_val = int(est["nnz"])
            rb = auto_row_block_rows(
                n_rows, est["estimated_peak_bytes"], memory_limit,
                user_row_block_rows=row_block_rows,
                total_nnz=total_nnz_val,
                zip_entry_limit=_ZIP_ENTRY_BYTES_LIMIT,
            )
            csv_record["row_block_rows"] = int(rb)
            csv_record["row_blocks"] = []
            log(
                f"SAVE {stage_name}: {csv_path} estimated_peak={est['estimated_peak_bytes']/1024**3:.2f}GB "
                f"> limit={memory_limit/1024**3:.2f}GB; saving row-sharded {output_format.upper()} NPZ, row_block_rows={rb}"
            )
            for ridx, r0 in enumerate(range(0, n_rows, rb)):
                r1 = min(n_rows, r0 + rb)
                mat = assemble_selected_matrix_for_csv(
                    manifest, selected_names, csv_path, output_format=output_format, row_slice=slice(r0, r1)
                )
                out = str(Path(output_dir) / f"{safe_stem(csv_path)}.{stage_name}.row{ridx:05d}.tog.{output_format}.npz")
                save_npz(out, mat, compressed=True)
                rec = {"path": out, "row_start": int(r0), "row_end": int(r1), "shape": [int(mat.shape[0]), int(mat.shape[1])], "nnz": int(mat.nnz)}
                csv_record["row_blocks"].append(rec)
                log(f"SAVE {stage_name}: {csv_path} rows={r0}:{r1} -> {out}, shape={mat.shape}, nnz={mat.nnz}")
                del mat
                gc.collect()
        snapshot["files"][csv_path] = csv_record
    return snapshot


def _level_rows_from_initial_tog_manifest(manifest, csv_path):
    for block in manifest.get("blocks", []):
        if csv_path in block.get("level_n_rows", {}):
            return int(block["level_n_rows"][csv_path])
        if csv_path in block.get("n_rows", {}):
            return int(block["n_rows"][csv_path]) + 1
    return 0


def load_initial_level_block_matrix(block_entry, csv_path, fmt="csr"):
    file_ref = block_entry.get("initial_level_files", {}).get(csv_path)
    ncols = len(block_entry.get("names", []))
    if file_ref is None:
        return sparse.csr_matrix((1, ncols), dtype=np.uint8).asformat(fmt)
    if str(file_ref).endswith(".npy"):
        arr = np.load(file_ref).astype(np.uint8, copy=False).reshape(1, -1)
        if arr.shape[1] != ncols:
            raise ValueError(f"initial level vector cols={arr.shape[1]} != block names={ncols}: {file_ref}")
        return sparse.csr_matrix(arr, dtype=np.uint8).asformat(fmt)
    mat = load_npz(file_ref)
    return mat.asformat(fmt)


def assemble_restored_level_matrix_for_csv(manifest, selected_names, csv_path, output_format="csr", row_slice=None):
    """Restore level rows from one initial level row plus TOG rows for one CSV.

    Level row i is initial XOR cumulative_xor(TOG rows [0, i)).  The function
    restores only row_slice when requested, so a long case can be emitted in
    bounded row shards after column cleaning.
    """
    selected_set = set(selected_names)
    name_order = {n: i for i, n in enumerate(selected_names)}
    n_level_rows = _level_rows_from_initial_tog_manifest(manifest, csv_path)
    if row_slice is None:
        row_start, row_end = 0, n_level_rows
    else:
        row_start = int(row_slice.start or 0)
        row_end = int(row_slice.stop if row_slice.stop is not None else n_level_rows)
        row_end = min(row_end, n_level_rows)
    row_start = max(0, min(row_start, n_level_rows))
    row_end = max(row_start, min(row_end, n_level_rows))

    parts = []
    for block in manifest["blocks"]:
        local_pairs = _selected_local_pairs_for_block(block, selected_set, name_order)
        if not local_pairs:
            continue
        local_idx = [p[0] for p in local_pairs]
        n_cols = len(local_idx)

        init = load_initial_level_block_matrix(block, csv_path, fmt="csr")[:, local_idx].tocsr()
        current = set(int(c) for c in init.indices[init.indptr[0]:init.indptr[1]])
        tog = load_cache_block_matrix(block, csv_path, fmt="csr")[:, local_idx].tocsr()
        tog_rows = int(tog.shape[0])

        prefix_end = min(row_start, tog_rows)
        for ridx in range(prefix_end):
            for col in tog.indices[tog.indptr[ridx]:tog.indptr[ridx + 1]]:
                col = int(col)
                if col in current:
                    current.remove(col)
                else:
                    current.add(col)

        out_indices = []
        out_indptr = [0]
        out_data = []
        for level_row in range(row_start, row_end):
            cols = sorted(current)
            out_indices.extend(cols)
            out_data.extend([1] * len(cols))
            out_indptr.append(len(out_indices))
            if level_row < row_end - 1 and level_row < tog_rows:
                for col in tog.indices[tog.indptr[level_row]:tog.indptr[level_row + 1]]:
                    col = int(col)
                    if col in current:
                        current.remove(col)
                    else:
                        current.add(col)

        part = sparse.csr_matrix(
            (np.asarray(out_data, dtype=np.uint8), np.asarray(out_indices, dtype=np.int64), np.asarray(out_indptr, dtype=np.int64)),
            shape=(row_end - row_start, n_cols),
            dtype=np.uint8,
        )
        parts.append(part.asformat(output_format))
        del init, tog, part, out_indices, out_indptr, out_data, current
        gc.collect()

    if not parts:
        return sparse.csr_matrix((row_end - row_start, len(selected_names)), dtype=np.uint8).asformat(output_format)
    out = sparse.hstack(parts, format=output_format, dtype=np.uint8)
    del parts
    gc.collect()
    return out


def estimate_restored_level_output_stats(manifest, selected_names, csv_path, output_format="csr"):
    n_rows = _level_rows_from_initial_tog_manifest(manifest, csv_path)
    n_cols = int(len(selected_names))
    # Restored level can be much denser than TOG.  Use a dense upper bound for
    # the guard so row sharding is conservative even without reading full level.
    dense_nnz = int(n_rows) * int(n_cols)
    est_bytes = estimate_sparse_matrix_bytes(n_rows, n_cols, dense_nnz, fmt=output_format, dtype_bytes=1)
    return {"n_rows": int(n_rows), "n_cols": int(n_cols), "nnz": int(dense_nnz), "estimated_peak_bytes": int(est_bytes)}


def save_restored_level_snapshot_memory_checked(
    manifest,
    selected_names,
    output_dir,
    stage_name,
    output_format="csr",
    max_mem_bytes=None,
    snapshot_mem_frac=0.60,
    row_block_rows=0,
    force_row_shard=False,
):
    """Save level snapshots restored from initial_level_files + TOG files."""
    mkdirp(output_dir)
    namefile = stage_namefile_path(output_dir, stage_name)
    write_lines(namefile, selected_names)

    if max_mem_bytes is None:
        max_mem_bytes = parse_mem_to_bytes("28GB")
    memory_limit = int(max_mem_bytes * float(snapshot_mem_frac))

    snapshot = OrderedDict()
    snapshot["stage"] = stage_name
    snapshot["namefile"] = namefile
    snapshot["cols"] = int(len(selected_names))
    snapshot["format"] = output_format
    snapshot["column_sharded"] = False
    snapshot["restored_from_initial_plus_tog"] = True
    snapshot["memory_limit_bytes"] = int(memory_limit)
    snapshot["files"] = OrderedDict()

    for csv_path in manifest["csv_paths"]:
        est = estimate_restored_level_output_stats(manifest, selected_names, csv_path, output_format=output_format)
        zip64_total_npz = _total_npz_bytes(est["n_rows"], est["n_cols"], est["nnz"], fmt=output_format, dtype_bytes=1)
        zip64_safe = zip64_total_npz <= _ZIP_ENTRY_BYTES_LIMIT
        est["zip64_total_npz_bytes"] = int(zip64_total_npz)
        est["zip64_safe"] = bool(zip64_safe)

        can_save_single = (
            (not force_row_shard)
            and (est["estimated_peak_bytes"] <= memory_limit)
            and zip64_safe
        )
        csv_record = OrderedDict()
        csv_record["estimated"] = est
        csv_record["single_file"] = bool(can_save_single)
        csv_record["row_sharded"] = not bool(can_save_single)

        if can_save_single:
            log(f"RESTORE LEVEL {stage_name}: {csv_path} saving single {output_format.upper()} NPZ")
            mat = assemble_restored_level_matrix_for_csv(manifest, selected_names, csv_path, output_format=output_format)
            out = str(Path(output_dir) / f"{safe_stem(csv_path)}.{stage_name}.tog.{output_format}.npz")
            save_npz(out, mat, compressed=True)
            csv_record["path"] = out
            csv_record["shape"] = [int(mat.shape[0]), int(mat.shape[1])]
            csv_record["nnz"] = int(mat.nnz)
            log(f"RESTORE LEVEL {stage_name}: {csv_path} -> {out}, shape={mat.shape}, nnz={mat.nnz}")
            del mat
            gc.collect()
        else:
            n_rows = int(est["n_rows"])
            rb = auto_row_block_rows(
                n_rows, est["estimated_peak_bytes"], memory_limit,
                user_row_block_rows=row_block_rows,
                total_nnz=int(est["nnz"]),
                zip_entry_limit=_ZIP_ENTRY_BYTES_LIMIT,
            )
            csv_record["row_block_rows"] = int(rb)
            csv_record["row_blocks"] = []
            log(
                f"RESTORE LEVEL {stage_name}: {csv_path} dense_guard_peak={est['estimated_peak_bytes']/1024**3:.2f}GB "
                f"> limit={memory_limit/1024**3:.2f}GB or ZIP unsafe; row-shard rows={rb}"
            )
            for ridx, r0 in enumerate(range(0, n_rows, rb)):
                r1 = min(n_rows, r0 + rb)
                mat = assemble_restored_level_matrix_for_csv(
                    manifest, selected_names, csv_path, output_format=output_format, row_slice=slice(r0, r1)
                )
                out = str(Path(output_dir) / f"{safe_stem(csv_path)}.{stage_name}.row{ridx:05d}.tog.{output_format}.npz")
                save_npz(out, mat, compressed=True)
                rec = {"path": out, "row_start": int(r0), "row_end": int(r1), "shape": [int(mat.shape[0]), int(mat.shape[1])], "nnz": int(mat.nnz)}
                csv_record["row_blocks"].append(rec)
                log(f"RESTORE LEVEL {stage_name}: {csv_path} rows={r0}:{r1} -> {out}, shape={mat.shape}, nnz={mat.nnz}")
                del mat
                gc.collect()
        snapshot["files"][csv_path] = csv_record
    return snapshot


# -----------------------------
# parse args and main
# -----------------------------
def parse_thresholds(s):
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty thresholds")
    return sorted(set(vals), reverse=True)


def main():
    ap = argparse.ArgumentParser(description="Huge-column precomputed-TOG CSV cleaner/reducer with one-pass CSV and block CSC cache")
    ap.add_argument("--filelist", required=False)
    ap.add_argument("--input-cache-manifest", default=None,
                    help="use an existing cache manifest (level or TOG) and skip CSV reading/cache generation")
    ap.add_argument("--input-tog-cache-manifest", default=None,
                    help="TOG cache manifest for level input resume (used with --input-data-type level + --input-cache-manifest)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-meta", required=True)
    ap.add_argument("--output-cols", required=True)
    ap.add_argument("--output-original-headers", required=True)
    ap.add_argument("--max-mem", default="28GB", help="memory budget, e.g. 28GB, 32000MB")
    ap.add_argument("--row-chunksize", type=int, default=0, help="0 = auto")
    ap.add_argument("--raw-col-block", type=int, default=0, help="0 = auto")
    ap.add_argument("--target-cols", type=int, default=100000)

    ap.add_argument("--cache-format", choices=["csc", "csr"], default="csc", help="internal TOG cache format; csc is recommended")
    ap.add_argument("--keep-cache", action="store_true", help="keep _work/tog_cache after completion")
    ap.add_argument("--final-format", choices=["csr", "csc"], default="csr", help="format for saved stage/final NPZ snapshots")
    ap.add_argument("--snapshot-mem-frac", type=float, default=0.60,
                    help="fraction of --max-mem allowed for assembling one complete per-CSV snapshot; exceed -> row sharding")
    ap.add_argument("--snapshot-row-block-rows", type=int, default=0,
                    help="row count per shard when snapshot exceeds memory check; 0 = auto")
    ap.add_argument("--force-row-shard-save", action="store_true",
                    help="always save stage/final snapshots by row shards; columns are still not split")

    ap.add_argument("--save-after-xz-clean", action="store_true")
    ap.add_argument("--save-after-zero", action="store_true")
    ap.add_argument("--save-after-one", action="store_true")
    ap.add_argument("--save-after-duplicate", action="store_true")
    ap.add_argument("--save-final", action="store_true")

    ap.add_argument("--disable-zero-remove", action="store_true")
    ap.add_argument("--disable-one-merge", action="store_true")
    ap.add_argument("--disable-duplicate-remove", action="store_true")
    ap.add_argument("--force-sim-reduce", action="store_true")
    ap.add_argument("--disable-sim-reduce", action="store_true")

    ap.add_argument("--similarity-thresholds", default="0.995,0.99,0.98,0.95,0.92,0.90")
    ap.add_argument("--num-perm", type=int, default=64)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-candidate-pairs", type=int, default=2000000)
    ap.add_argument("--exact-dup-group-limit", type=int, default=20000)
    ap.add_argument("--duplicate-mode", choices=["hash-only", "exact"], default="hash-only",
                    help="hash-only is fastest; exact performs sparse CSC exact verification inside digest buckets")
    ap.add_argument("--duplicate-exact-strategy", choices=["one-pass", "bucket"], default="one-pass",
                    help="one-pass scans cache blocks once for all non-singleton buckets and is usually faster; bucket verifies one digest bucket at a time and uses less memory")
    ap.add_argument("--duplicate-progress-interval", type=int, default=200000)
    ap.add_argument("--duplicate-verbose", action="store_true",
                    help="print per-cache-file duplicate verification logs; useful for locating IO bottlenecks")
    ap.add_argument("--input-data-type", choices=["tog", "level"], default="tog",
                    help="input data type: 'tog' (default, assumes TOG CSV) or 'level' (produces both level and TOG outputs)")
    args = ap.parse_args()

    t_all = time.time()
    mkdirp(args.output_dir)
    work_dir = Path(args.output_dir) / "_work"
    cache_dir = work_dir / "tog_cache"
    mkdirp(work_dir)
    mkdirp(cache_dir)
    stage_snapshots = OrderedDict()

    layout_jsonl = str(work_dir / "expanded_layout.jsonl")
    stats_jsonl = str(work_dir / "column_stats_after_xz.jsonl")
    manifest_json = str(work_dir / "tog_cache_manifest.json")

    max_mem_bytes = parse_mem_to_bytes(args.max_mem)

    if args.input_cache_manifest:
        manifest_json = os.path.abspath(args.input_cache_manifest)
        manifest = load_manifest(manifest_json)
        csv_paths = list(manifest.get("csv_paths", []))
        if not csv_paths:
            raise ValueError(f"input cache manifest has empty csv_paths: {manifest_json}")
        args.cache_format = manifest.get("cache_format", args.cache_format)
        source_headers = Path(manifest_json).parent.parent / "original_headers.txt"
        if source_headers.is_file():
            with open(source_headers, "r", encoding="utf-8") as f:
                header = [line.rstrip("\n") for line in f]
            write_lines(args.output_original_headers, header)
        else:
            header = ["time"]
            write_lines(args.output_original_headers, header)
        expanded_names = [
            name
            for block in manifest.get("blocks", [])
            for name in list(block.get("names", [])) + list(block.get("removed_xz_names", []))
        ]
        xz_removed = [
            name
            for block in manifest.get("blocks", [])
            for name in block.get("removed_xz_names", [])
        ]
        row_chunksize = int(manifest.get("row_chunksize", args.row_chunksize or 0))
        raw_col_block = int(manifest.get("raw_col_block", args.raw_col_block or 0))
        layout_jsonl = str(Path(manifest_json).parent / "expanded_layout.jsonl")

        # Load TOG manifest for level data resume
        tog_manifest = None
        if args.input_data_type == "level":
            if args.input_tog_cache_manifest:
                tog_manifest_json = os.path.abspath(args.input_tog_cache_manifest)
            else:
                # Try auto-locate
                for candidate in [
                    str(Path(manifest_json).parent / "tog_cache_manifest.json"),
                    str(Path(manifest_json).with_name("tog_cache_manifest.json")),
                ]:
                    if os.path.isfile(candidate):
                        tog_manifest_json = candidate
                        break
                else:
                    raise ValueError(
                        "level input with --input-cache-manifest requires --input-tog-cache-manifest "
                        "or a tog_cache_manifest.json in the same directory"
                    )
            tog_manifest = load_manifest(tog_manifest_json)
            log(f"LOADED TOG manifest: {tog_manifest_json}")
            log(f"TOG input_files={len(tog_manifest.get('csv_paths', []))}")

        log("=" * 80)
        log(f"USE EXISTING CACHE MANIFEST: {manifest_json}")
        log(f"input_files={len(csv_paths)}")
        log(f"cache_format={args.cache_format}")
        log(f"expanded_cols_before_xz≈{len(expanded_names)}")
        log(f"xz_removed_count={len(xz_removed)}")
    else:
        if not args.filelist:
            raise ValueError("--filelist is required unless --input-cache-manifest is used")
        csv_paths = read_filelist(args.filelist)
        header = ensure_same_header_streaming(csv_paths)
        sample_row = read_first_data_row(csv_paths[0])
        write_lines(args.output_original_headers, header)

        layout, expanded_names, raw_to_records = build_expanded_layout(header, sample_row, layout_jsonl)
        avg_width = max(1.0, len(layout) / max(1, len(header) - 1))
        row_chunksize, raw_col_block = auto_sizes(max_mem_bytes, avg_width, args.row_chunksize, args.raw_col_block)

        log("=" * 80)
        log("REMINDER: 输入CSV已按TOG数据处理，本脚本不会再执行level->tog XOR diff，也不会因diff丢弃首行。")
        log(f"input_files={len(csv_paths)}")
        log(f"raw_cols_after_drop_time={len(header) - 1}")
        log(f"expanded_cols={len(expanded_names)} avg_width={avg_width:.3f}")
        log(f"max_mem={args.max_mem} -> row_chunksize={row_chunksize}, raw_col_block={raw_col_block}")
        log(f"cache_format={args.cache_format}, cache_dir={cache_dir}")

        all_raw_indices = list(range(1, len(header)))
        #newmodify zhangyu
        use_fast_bit_reader = is_fast_bit_layout(layout)
        if use_fast_bit_reader:
            log("USE FAST BIT CSV READER: bypass pandas.read_csv for bit-level CSV")
        else:
            log("USE ORIGINAL PANDAS CSV READER: non-bit-level layout detected")

        result = build_tog_cache_once(
            csv_paths=csv_paths,
            raw_to_records=raw_to_records,
            all_raw_indices=all_raw_indices,
            row_chunksize=row_chunksize,
            raw_col_block=raw_col_block,
            cache_dir=str(cache_dir),
            cache_format=args.cache_format,
            use_fast_bit_reader=use_fast_bit_reader,
            total_signal_cols=len(header) - 1,
            input_data_type=args.input_data_type,
        )
        if args.input_data_type == "level":
            manifest, xz_removed, tog_manifest = result
        else:
            manifest, xz_removed = result
            tog_manifest = None
        save_manifest(manifest_json, manifest)
        if tog_manifest is not None:
            tog_manifest_json = str(work_dir / "tog_cache_manifest.json")
            level_manifest_json = str(work_dir / "level_cache_manifest.json")
            save_manifest(tog_manifest_json, tog_manifest)
            save_manifest(level_manifest_json, manifest)
            log(f"SAVED LEVEL manifest: {level_manifest_json}")
            log(f"SAVED TOG manifest: {tog_manifest_json}")

    # ================================================================
    # Pipeline: collect column sets from TOG, then apply to both TOG and level
    # ================================================================
    def collect_column_sets_from_manifest(pipeline_manifest, pipeline_stats_jsonl, pipeline_label=""):
        """Run reduction pipeline on a manifest to determine which columns to keep.
        Only collects column name lists — does NOT save stage snapshots.
        Returns dict: {stage: (col_names_list, extra_info_dict)}.
        """
        pipeline_stats = collect_stats_from_cache(pipeline_manifest, pipeline_stats_jsonl)
        pipeline_stage = build_stage_columns(pipeline_stats, keep_all_one_one_col=not args.disable_one_merge)

        col_sets = {}
        col_sets["stats"] = pipeline_stats
        col_sets["manifest"] = pipeline_manifest

        # XZ clean
        cols_xz = pipeline_stage["after_xz_clean"]
        col_sets["after_xz_clean"] = {"cols": cols_xz}

        # Zero remove
        if args.disable_zero_remove:
            cols_zero = list(cols_xz)
            zero_removed_list = []
        else:
            zero_removed_set = set(pipeline_stage["zero_removed"])
            cols_zero = [c for c in cols_xz if c not in zero_removed_set]
            zero_removed_list = pipeline_stage["zero_removed"]
        col_sets["after_zero"] = {"cols": cols_zero, "zero_removed": zero_removed_list}

        # One merge
        if args.disable_one_merge:
            cols_one = list(cols_zero)
            kept_one_sig = None
            removed_one_list = []
        else:
            removed_one_set = set(pipeline_stage["removed_all_one_cols"])
            cols_one = [c for c in cols_zero if c not in removed_one_set]
            kept_one_sig = pipeline_stage["kept_all_one_col"] if pipeline_stage["kept_all_one_col"] in cols_zero else None
            removed_one_list = [c for c in pipeline_stage["removed_all_one_cols"] if c in cols_zero]
        col_sets["after_one"] = {"cols": cols_one, "kept_all_one": kept_one_sig, "removed_all_one": removed_one_list}

        # Duplicate remove
        if args.disable_duplicate_remove:
            cols_dup = list(cols_one)
            dup_map = OrderedDict()
            large_dup = []
        else:
            log(f"[{pipeline_label}] START duplicate removal: mode={args.duplicate_mode}")
            if args.duplicate_mode == "hash-only":
                cols_dup, dup_map, large_dup = remove_duplicates_by_hash(
                    pipeline_stats, cols_one, args.duplicate_progress_interval
                )
            else:
                cols_dup, dup_map, large_dup = remove_exact_duplicates(
                    pipeline_manifest, pipeline_stats, cols_one,
                    exact_group_limit=args.exact_dup_group_limit,
                    progress_interval=args.duplicate_progress_interval,
                    strategy=args.duplicate_exact_strategy,
                    verbose=args.duplicate_verbose,
                )
            log(f"[{pipeline_label}] DUP done: before={len(cols_one)}, after={len(cols_dup)}, removed={len(dup_map)}")
        col_sets["after_duplicate"] = {"cols": cols_dup, "dup_map": dup_map, "large_dup": large_dup}

        # Similarity reduce
        thresholds = parse_thresholds(args.similarity_thresholds)
        do_sim = (args.force_sim_reduce or len(cols_dup) > args.target_cols) and not args.disable_sim_reduce
        sim_drop = OrderedDict()
        sim_reports_list = []
        if do_sim:
            log(f"[{pipeline_label}] START progressive similarity reduction")
            final_cols_list, sim_drop, sim_reports_list = progressive_similarity_reduce(
                pipeline_manifest, cols_dup, args.target_cols, thresholds,
                args.num_perm, args.bands, args.seed, args.max_candidate_pairs
            )
        else:
            final_cols_list = list(cols_dup)
        col_sets["final"] = {"cols": final_cols_list, "sim_drop": sim_drop, "sim_reports": sim_reports_list}

        # Summary info
        col_sets["summary"] = {
            "label": pipeline_label,
            "xz_removed_count": pipeline_manifest.get("xz_removed_count", 0),
            "zero_removed_count": len(zero_removed_list),
            "one_merged_count": len(removed_one_list),
            "kept_all_one_col": kept_one_sig,
            "duplicate_removed_count": len(dup_map),
            "similarity_removed_count": len(sim_drop),
            "similarity_enabled": do_sim,
            "similarity_reports": sim_reports_list,
            "counts": {
                "after_xz_clean": len(cols_xz),
                "after_zero": len(cols_zero),
                "after_one": len(cols_one),
                "after_duplicate": len(cols_dup),
                "final": len(final_cols_list),
                "target_cols": args.target_cols,
            },
        }
        return col_sets


    def save_stages_from_column_sets(
        col_sets,
        output_manifest,
        output_dir,
        output_label="",
        output_format=None,
    ):
        """Save stage snapshots using PRE-COMPUTED column selections from col_sets.
        This allows applying the same column selections (computed from TOG) to level data.
        Returns snapshots_dict.
        """
        fmt = output_format or args.final_format
        snapshots = OrderedDict()

        for stage_name in ["after_xz_clean", "after_zero", "after_one", "after_duplicate", "final"]:
            stage_info = col_sets.get(stage_name)
            if stage_info is None:
                continue
            cols_list = stage_info.get("cols", [])
            if not cols_list:
                continue

            should_save = (
                (stage_name == "after_xz_clean" and args.save_after_xz_clean)
                or (stage_name == "after_zero" and args.save_after_zero)
                or (stage_name == "after_one" and args.save_after_one)
                or (stage_name == "after_duplicate" and args.save_after_duplicate)
                or (stage_name == "final" and args.save_final)
            )
            if not should_save:
                continue

            log(f"[{output_label}] SAVING stage '{stage_name}': {len(cols_list)} cols → {output_dir}")
            snapshot = save_stage_snapshot_memory_checked(
                output_manifest, cols_list, output_dir, stage_name, fmt,
                max_mem_bytes=max_mem_bytes, snapshot_mem_frac=args.snapshot_mem_frac,
                row_block_rows=args.snapshot_row_block_rows, force_row_shard=args.force_row_shard_save,
            )
            snapshots[stage_name] = snapshot

        # Write cols file for final stage
        final_cols_list = col_sets.get("final", {}).get("cols", [])
        if final_cols_list:
            final_cols_path = str(Path(output_dir).parent / f"{output_label}_final_cols.txt") if output_label else args.output_cols
            write_lines(final_cols_path, final_cols_list)

        return snapshots

    # ================================================================
    # Run pipeline(s)
    # ================================================================
    if args.input_data_type == "level":
        # 1. Run pipeline on TOG manifest to determine column selections
        tog_stats_jsonl = str(work_dir / "tog_column_stats.jsonl")
        log("=" * 60)
        log("STEP 1/2: Running pipeline on TOG data to determine column selections")
        log("=" * 60)
        tog_col_sets = collect_column_sets_from_manifest(tog_manifest, tog_stats_jsonl, pipeline_label="tog")

        # 2. Save TOG outputs using TOG column selections
        tog_output_dir = str(Path(args.output_dir) / "tog")
        mkdirp(tog_output_dir)
        log("")
        log("=" * 60)
        log("STEP 2a/2: Saving TOG outputs with TOG column selections")
        log("=" * 60)
        tog_snapshots = save_stages_from_column_sets(
            tog_col_sets, tog_manifest, tog_output_dir, output_label="tog"
        )

        # 3. Save level outputs using the SAME column selections from TOG
        level_output_dir = str(Path(args.output_dir) / "level")
        mkdirp(level_output_dir)
        log("")
        log("=" * 60)
        log("STEP 2b/2: Saving level outputs with SAME column selections (from TOG)")
        log("=" * 60)
        use_restored_level = (
            tog_manifest.get("level_cache_mode") == "initial_plus_tog"
            or any(block.get("initial_level_files") for block in tog_manifest.get("blocks", []))
        )
        if use_restored_level:
            log("LEVEL restore mode: using initial_level_files + TOG after column cleaning")

            def save_restored_level_stages_from_column_sets(col_sets, output_dir, output_label="level", output_format=None):
                fmt = output_format or args.final_format
                snapshots = OrderedDict()
                for stage_name in ["after_xz_clean", "after_zero", "after_one", "after_duplicate", "final"]:
                    stage_info = col_sets.get(stage_name)
                    if stage_info is None:
                        continue
                    cols_list = stage_info.get("cols", [])
                    if not cols_list:
                        continue
                    should_save = (
                        (stage_name == "after_xz_clean" and args.save_after_xz_clean)
                        or (stage_name == "after_zero" and args.save_after_zero)
                        or (stage_name == "after_one" and args.save_after_one)
                        or (stage_name == "after_duplicate" and args.save_after_duplicate)
                        or (stage_name == "final" and args.save_final)
                    )
                    if not should_save:
                        continue
                    log(f"[{output_label}] RESTORING stage '{stage_name}': {len(cols_list)} cols -> {output_dir}")
                    snapshots[stage_name] = save_restored_level_snapshot_memory_checked(
                        tog_manifest, cols_list, output_dir, stage_name, fmt,
                        max_mem_bytes=max_mem_bytes, snapshot_mem_frac=args.snapshot_mem_frac,
                        row_block_rows=args.snapshot_row_block_rows, force_row_shard=args.force_row_shard_save,
                    )
                final_cols_list = col_sets.get("final", {}).get("cols", [])
                if final_cols_list:
                    write_lines(str(Path(output_dir).parent / f"{output_label}_final_cols.txt"), final_cols_list)
                return snapshots

            level_snapshots = save_restored_level_stages_from_column_sets(
                tog_col_sets, level_output_dir, output_label="level"
            )
        else:
            level_snapshots = save_stages_from_column_sets(
                tog_col_sets, manifest, level_output_dir, output_label="level"
            )

        # Build combined metadata
        stage_snapshots = OrderedDict()
        stage_snapshots["level"] = level_snapshots
        stage_snapshots["tog"] = tog_snapshots

        tog_summary = tog_col_sets["summary"]
        meta = OrderedDict()
        meta["input_files"] = csv_paths
        meta["input_data_type"] = "level"
        meta["column_selection_source"] = "TOG data; same column indices applied to level output"
        meta["original_header_count"] = len(header)
        meta["raw_cols_after_drop_time"] = len(header) - 1
        meta["expanded_cols_before_xz"] = len(expanded_names)
        meta["tog_note"] = "TOG = Level[t+1] XOR Level[t]; first row dropped. Level output retains all rows."
        meta["level_restore_mode"] = "initial_plus_tog" if use_restored_level else "full_level_cache"
        meta["counts"] = tog_summary["counts"]
        meta["xz_removed_count"] = tog_summary["xz_removed_count"]
        meta["zero_removed_count"] = tog_summary["zero_removed_count"]
        meta["one_merged_count"] = tog_summary["one_merged_count"]
        meta["duplicate_removed_count"] = tog_summary["duplicate_removed_count"]
        meta["similarity_removed_count"] = tog_summary["similarity_removed_count"]
        meta["total_seconds"] = time.time() - t_all
    else:
        # TOG-only mode (original behavior)
        stats_jsonl_path = str(work_dir / "column_stats_after_xz.jsonl")
        tog_col_sets = collect_column_sets_from_manifest(manifest, stats_jsonl_path, pipeline_label="tog")
        stage_snapshots = save_stages_from_column_sets(
            tog_col_sets, manifest, args.output_dir, output_label=""
        )
        tog_summary = tog_col_sets["summary"]
        meta = OrderedDict()
        meta["input_files"] = csv_paths
        meta["input_data_type"] = "tog"
        meta["original_header_count"] = len(header)
        meta["raw_cols_after_drop_time"] = len(header) - 1
        meta["expanded_cols_before_xz"] = len(expanded_names)
        meta["tog_diff_applied"] = False
        meta["diff_note"] = "Input CSV assumed to be TOG data; no XOR diff applied."
        meta["counts"] = tog_summary["counts"]
        meta["xz_removed_count"] = tog_summary["xz_removed_count"]
        meta["zero_removed_count"] = tog_summary["zero_removed_count"]
        meta["one_merged_count"] = tog_summary["one_merged_count"]
        meta["duplicate_removed_count"] = tog_summary["duplicate_removed_count"]
        meta["similarity_removed_count"] = tog_summary["similarity_removed_count"]
        meta["total_seconds"] = time.time() - t_all

    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if not args.keep_cache:
        log(f"REMOVE temporary cache dir: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    log("=" * 80)
    log("DONE")
    if args.input_data_type == "level":
        log(f"level counts={meta.get('level', {}).get('counts', {})}")
        log(f"tog counts={meta.get('tog', {}).get('counts', {})}")
    else:
        log(f"counts={meta.get('counts', {})}")
    log(f"output_cols={args.output_cols}")
    log(f"output_meta={args.output_meta}")
    log(f"total_seconds={meta['total_seconds']:.2f}")


if __name__ == "__main__":
    main()
