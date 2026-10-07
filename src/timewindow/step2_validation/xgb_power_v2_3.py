# -*- coding: UTF-8 -*-
"""
Step2 后的 XGBoost 功耗预测验证（由 old-version/xgb_power_v2_3.py 复制适配）。

数据接口：
1. 从 Step1 manifest 取得 combo 顺序、train/test stems 及各 combo 文件列表。
2. 从 Step2 manifest 取得每个 combo 的 selected_indices_file。
3. 每个 case 按 combo 执行列选择并横向拼接，作为本脚本的输入特征。
4. train stems 仅用于 train/val；test stems 全部固定为测试集。
5. XGBoost 训练参数、训练过程及评估体系保持原脚本默认设置。
"""

from __future__ import print_function

import argparse
import io
import json
import zipfile
import os
import re
import shutil
import time
from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, load_npz, save_npz, vstack, hstack

from xgb_power_eval_from_prediction_csv import evaluate_prediction_csv
from pipeline_combo_dataset import build_pipeline_chunks


# -----------------------------
# Basic utilities
# -----------------------------

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def read_nonempty_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def read_feature_filelist(path):
    pairs = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    "feature-filelist line %d must contain two files: <level.npz> <tog.npz>, got: %s"
                    % (lineno, line)
                )
            pairs.append((parts[0], parts[1]))
    return pairs


def load_pair_lists(label_filelist, feature_filelist):
    label_files = read_nonempty_lines(label_filelist)
    feature_pairs = read_feature_filelist(feature_filelist)
    if len(label_files) != len(feature_pairs):
        raise ValueError(
            "label-filelist count (%d) != feature-filelist count (%d)"
            % (len(label_files), len(feature_pairs))
        )
    return label_files, feature_pairs


def file_signature(paths):
    sig = []
    for p in paths:
        st = os.stat(p)
        sig.append({
            "path": os.path.abspath(p),
            "size": int(st.st_size),
            "mtime": float(st.st_mtime),
        })
    return sig


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def meta_equal(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def parse_top_ns(s):
    if s is None or str(s).strip() == "":
        return [512]
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out if out else [512]


def dtype_from_string(s):
    if s == "float64":
        return np.float64
    return np.float32


# -----------------------------
# Alignment and sparse construction
# -----------------------------

def valid_sample_range(label_len, level_rows, tog_rows,
                       seq_len, before_cycles, label_start_cycle):
    """
    label index i 的 label cycle = label_start_cycle + i。
    row_start = label_cycle - seq_len - before_cycles + 1。
    返回有效 label index 和对应 feature row_start。
    """
    idx = np.arange(label_len, dtype=np.int64)
    label_cycles = int(label_start_cycle) + idx
    row_start = label_cycles - int(seq_len) - int(before_cycles) + 1
    row_end = row_start + int(seq_len)
    max_rows = min(int(level_rows), int(tog_rows))
    mask = (row_start >= 0) & (row_end <= max_rows)
    return idx[mask], row_start[mask]


def build_feature_block_no_topk(level_csr, tog_csr, row_start_ids, seq_len,
                                data_dtype=np.float32):
    """
    关闭 TopK 的 block 构建。

    对于一个 block 中的样本，批量取：
      level[row_start + off, :] for off in 0..seq_len-1
      tog[row_start + off, :]   for off in 0..seq_len-1

    然后按列 hstack：
      [level_off0, ..., level_off{seq_len-1}, tog_off0, ..., tog_off{seq_len-1}]

    注意：
    - 不做 toarray
    - 不逐样本 hstack
    - 仍然会产生 shape=(block_samples, raw_ncols*seq_len*2) 的超宽 CSR
    """
    n_samples = int(row_start_ids.size)
    raw_ncols = int(level_csr.shape[1])
    total_cols = raw_ncols * int(seq_len) * 2

    if n_samples == 0:
        return csr_matrix((0, total_cols), dtype=data_dtype)

    blocks = []

    for off in range(int(seq_len)):
        rows = row_start_ids + off
        sub = level_csr[rows, :].tocsr()
        if sub.data.dtype != data_dtype:
            sub.data = sub.data.astype(data_dtype, copy=False)
        blocks.append(sub)

    for off in range(int(seq_len)):
        rows = row_start_ids + off
        sub = tog_csr[rows, :].tocsr()
        if sub.data.dtype != data_dtype:
            sub.data = sub.data.astype(data_dtype, copy=False)
        blocks.append(sub)

    return hstack(blocks, format="csr", dtype=data_dtype)


# -----------------------------
# Cache
# -----------------------------

def chunk_paths(cache_dir):
    chunk_dir = os.path.join(cache_dir, "chunks")
    meta_file = os.path.join(cache_dir, "dataset.meta.json")
    return chunk_dir, meta_file


def list_chunk_files(chunk_dir):
    xs = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith("_X.npz")])
    ys = [x[:-6] + "_y.npy" for x in xs]
    for y in ys:
        if not os.path.exists(y):
            raise RuntimeError("missing label chunk: %s" % y)
    return xs, ys


def list_chunk_meta_files(chunk_x_files):
    """
    每个 X chunk 对应一个 _m.npy 元信息文件：
      col0=file_idx, col1=label_idx, col2=label_cycle
    旧 cache 可能没有该文件；这种情况下返回 None。
    """
    ms = [x[:-6] + "_m.npy" for x in chunk_x_files]
    if all(os.path.exists(m) for m in ms):
        return ms
    return None


def clear_chunk_dir(chunk_dir):
    ensure_dir(chunk_dir)
    for f in os.listdir(chunk_dir):
        p = os.path.join(chunk_dir, f)
        if os.path.isfile(p):
            os.remove(p)


def sparse_npz_shape(path):
    """Read only the tiny shape.npy member instead of decompressing the CSR arrays."""
    with zipfile.ZipFile(path, "r") as zf:
        shape_data = zf.read("shape.npy")
    shape = np.load(io.BytesIO(shape_data), allow_pickle=False)
    return tuple(int(x) for x in shape)


def infer_and_check_raw_ncols(feature_pairs):
    raw_ncols = None
    for level_f, tog_f in feature_pairs:
        level_shape = sparse_npz_shape(level_f)
        tog_shape = sparse_npz_shape(tog_f)
        if level_shape[1] != tog_shape[1]:
            raise ValueError(
                "level/tog raw cols mismatch: %s has %d, %s has %d"
                % (level_f, level_shape[1], tog_f, tog_shape[1])
            )
        if raw_ncols is None:
            raw_ncols = int(level_shape[1])
        elif int(level_shape[1]) != raw_ncols:
            raise ValueError(
                "raw_ncols mismatch across files: first=%d, %s has %d"
                % (raw_ncols, level_f, level_shape[1])
            )
    if raw_ncols is None:
        raise ValueError("empty feature filelist")
    return raw_ncols


def dataset_meta(label_files, feature_pairs, raw_ncols, args):
    all_paths = list(label_files)
    for level_f, tog_f in feature_pairs:
        all_paths.append(level_f)
        all_paths.append(tog_f)

    return {
        "version": 101,
        "mode": "no_topk_all_raw_columns",
        "raw_ncols": int(raw_ncols),
        "expanded_ncols": int(raw_ncols) * int(args.length) * 2,
        "seq_len": int(args.length),
        "before_cycles": int(args.before_cycles),
        "label_start_cycle": int(args.label_start_cycle),
        "label_drop_first": False,
        "label_scale": float(args.label_scale),
        "block_size": int(args.build_block_size),
        "data_dtype": str(args.data_dtype),
        "files": file_signature(all_paths),
    }


def cache_valid(cache_dir, expected_meta):
    chunk_dir, meta_file = chunk_paths(cache_dir)
    if not os.path.exists(meta_file) or not os.path.isdir(chunk_dir):
        return False
    try:
        old_meta = load_json(meta_file)
    except Exception:
        return False
    if not meta_equal(old_meta, expected_meta):
        return False
    xs, ys = list_chunk_files(chunk_dir)
    return len(xs) > 0


def build_dataset_chunks_no_topk(label_files, feature_pairs, raw_ncols, args):
    """
    每个输入文件独立处理，文件之间不会跨窗口。
    按 block 保存 X.npz + y.npy，避免构建阶段一次性持有全量展开矩阵。
    """
    ensure_dir(args.cache_dir)
    chunk_dir, meta_file = chunk_paths(args.cache_dir)
    ensure_dir(chunk_dir)

    meta = dataset_meta(label_files, feature_pairs, raw_ncols, args)

    if (not args.force_rebuild_dataset) and cache_valid(args.cache_dir, meta):
        print("[CACHE] dataset chunks are valid, skip rebuild:", chunk_dir)
        return list_chunk_files(chunk_dir)

    print("[BUILD] rebuild dataset chunks:", chunk_dir)
    clear_chunk_dir(chunk_dir)

    seq_len = int(args.length)
    before_cycles = int(args.before_cycles)
    label_start_cycle = int(args.label_start_cycle)
    block_size = int(args.build_block_size)
    label_scale = float(args.label_scale)
    data_dtype = dtype_from_string(args.data_dtype)

    chunk_id = 0
    total_samples = 0
    total_nnz = 0

    for file_idx, (label_f, pair) in enumerate(zip(label_files, feature_pairs)):
        level_f, tog_f = pair
        t_file = time.time()

        print("[FILE] %d/%d" % (file_idx + 1, len(label_files)))
        print("       label:", label_f)
        print("       level:", level_f)
        print("       tog  :", tog_f)

        label_raw = np.load(label_f)
        if label_raw.shape[0] <= 0:
            print("       skip: label length <= 0")
            continue

        # v2_3: label 数据已提前处理，不再 drop 第一行。
        label = label_raw.astype(np.float32, copy=False) / label_scale

        level = load_npz(level_f).tocsr()
        tog = load_npz(tog_f).tocsr()

        if level.shape[1] != raw_ncols or tog.shape[1] != raw_ncols:
            raise ValueError("raw_ncols mismatch while building dataset")
        if level.shape[1] != tog.shape[1]:
            raise ValueError("level/tog raw cols mismatch")

        valid_label_idx, row_start = valid_sample_range(
            label_len=label.shape[0],
            level_rows=level.shape[0],
            tog_rows=tog.shape[0],
            seq_len=seq_len,
            before_cycles=before_cycles,
            label_start_cycle=label_start_cycle,
        )

        if valid_label_idx.size == 0:
            print("       no valid samples after alignment")
            del label_raw, label, level, tog
            continue

        print("       valid samples:", valid_label_idx.size)

        for b0 in range(0, valid_label_idx.size, block_size):
            b1 = min(b0 + block_size, valid_label_idx.size)
            rows = row_start[b0:b1]
            y = label[valid_label_idx[b0:b1]].reshape(-1, 1)
            label_idx_block = valid_label_idx[b0:b1].astype(np.int64, copy=False)
            label_cycle_block = (int(label_start_cycle) + label_idx_block).astype(np.int64, copy=False)
            sample_meta = np.column_stack([
                np.full(label_idx_block.shape, int(file_idx), dtype=np.int64),
                label_idx_block,
                label_cycle_block,
            ])

            t0 = time.time()
            X_block = build_feature_block_no_topk(
                level_csr=level,
                tog_csr=tog,
                row_start_ids=rows,
                seq_len=seq_len,
                data_dtype=data_dtype,
            )

            x_file = os.path.join(chunk_dir, "part_%06d_X.npz" % chunk_id)
            y_file = os.path.join(chunk_dir, "part_%06d_y.npy" % chunk_id)
            m_file = os.path.join(chunk_dir, "part_%06d_m.npy" % chunk_id)

            save_npz(x_file, X_block)
            np.save(y_file, y)
            np.save(m_file, sample_meta)

            total_samples += int(X_block.shape[0])
            total_nnz += int(X_block.nnz)

            print("       chunk %06d: X=%s nnz=%d y=%s time=%.2fs" % (
                chunk_id, str(X_block.shape), X_block.nnz, str(y.shape), time.time() - t0
            ))

            chunk_id += 1
            del X_block, y, sample_meta

        del label_raw, label, level, tog
        print("       file done, elapsed %.2fs" % (time.time() - t_file))

    meta["total_samples"] = int(total_samples)
    meta["total_nnz"] = int(total_nnz)
    meta["num_chunks"] = int(chunk_id)
    save_json(meta_file, meta)

    print("[BUILD] dataset done: samples=%d, nnz=%d, chunks=%d" % (
        total_samples, total_nnz, chunk_id
    ))

    return list_chunk_files(chunk_dir)


def count_chunk_rows(chunk_y_files):
    """只读取 y.npy 的 shape，统计总样本数，不加载 X。"""
    rows = []
    total = 0
    for yf in chunk_y_files:
        y = np.load(yf, mmap_mode="r")
        n = int(y.shape[0])
        rows.append(n)
        total += n
    return rows, total


class CSRChunkDataIter(xgb.DataIter):
    """
    XGBoost 迭代式数据加载器。

    每次 next() 只加载一个 cache chunk 中与 [row_begin, row_end) 相交的部分，
    通过 input_data(data=csr, label=y) 交给 XGBoost 构建 QuantileDMatrix。

    这样训练入口不再执行：
      1) 所有 chunk -> Python list
      2) scipy.sparse.vstack -> 全量大 CSR
      3) xgb.DMatrix -> 再复制/量化
    可以显著降低 Python 侧峰值内存。
    """
    def __init__(self, chunk_x_files, chunk_y_files, row_begin, row_end, name="train", row_indices=None):
        self.chunk_x_files = list(chunk_x_files)
        self.chunk_y_files = list(chunk_y_files)
        self.row_begin = int(row_begin)
        self.row_end = int(row_end)
        self.name = str(name)
        self.row_indices = None if row_indices is None else np.sort(np.asarray(row_indices, dtype=np.int64))

        if len(self.chunk_x_files) != len(self.chunk_y_files):
            raise ValueError("chunk_x_files count != chunk_y_files count")
        if self.row_begin < 0 or self.row_end < self.row_begin:
            raise ValueError("invalid row range: [%d, %d)" % (self.row_begin, self.row_end))

        self.chunk_rows, self.total_rows = count_chunk_rows(self.chunk_y_files)
        if self.row_end > self.total_rows:
            raise ValueError(
                "row_end=%d exceeds total rows=%d" % (self.row_end, self.total_rows)
            )
        if self.row_indices is not None:
            if self.row_indices.size == 0:
                raise ValueError("row_indices is empty for %s" % self.name)
            if int(self.row_indices[0]) < 0 or int(self.row_indices[-1]) >= self.total_rows:
                raise ValueError("row_indices out of range for %s" % self.name)

        # prefix sum: global row range of each chunk is [chunk_offsets[i], chunk_offsets[i+1])
        self.chunk_offsets = np.zeros(len(self.chunk_rows) + 1, dtype=np.int64)
        if self.chunk_rows:
            self.chunk_offsets[1:] = np.cumsum(np.asarray(self.chunk_rows, dtype=np.int64))

        self._cursor = 0
        self._yielded_rows = 0

        try:
            super(CSRChunkDataIter, self).__init__()
        except TypeError:
            # 兼容部分旧版本 xgboost.DataIter 构造函数差异。
            pass

    def reset(self):
        self._cursor = 0
        self._yielded_rows = 0

    def next(self, input_data):
        while self._cursor < len(self.chunk_x_files):
            i = self._cursor
            self._cursor += 1

            c0 = int(self.chunk_offsets[i])
            c1 = int(self.chunk_offsets[i + 1])

            t0 = time.time()

            if self.row_indices is None:
                # 当前 chunk 与目标 row range 没有交集。
                if c1 <= self.row_begin or c0 >= self.row_end:
                    continue

                # 交集映射到 chunk 内部 slice。
                s0 = max(self.row_begin, c0) - c0
                s1 = min(self.row_end, c1) - c0

                X = load_npz(self.chunk_x_files[i]).tocsr()
                y = np.load(self.chunk_y_files[i]).reshape(-1)

                if s0 != 0 or s1 != X.shape[0]:
                    X = X[int(s0):int(s1), :].tocsr()
                    y = y[int(s0):int(s1)]
            else:
                # shuffle split 后，当前 split 对应的是一组全局 row index。
                left = np.searchsorted(self.row_indices, c0, side="left")
                right = np.searchsorted(self.row_indices, c1, side="left")
                if right <= left:
                    continue

                local_rows = self.row_indices[left:right] - c0
                X_full = load_npz(self.chunk_x_files[i]).tocsr()
                y_full = np.load(self.chunk_y_files[i]).reshape(-1)
                X = X_full[local_rows, :].tocsr()
                y = y_full[local_rows]
                del X_full, y_full

            self._yielded_rows += int(X.shape[0])
            print("[ITER-%s] chunk %d/%d rows=%d X=%s nnz=%d time=%.2fs" % (
                self.name, i + 1, len(self.chunk_x_files), X.shape[0],
                str(X.shape), X.nnz, time.time() - t0
            ))

            input_data(data=X, label=y)
            return 1

        return 0


def make_quantile_dmatrix_from_chunks(chunk_x_files, chunk_y_files, row_begin, row_end,
                                      name, max_bin, ref=None, row_indices=None):
    """
    使用 DataIter 构建 QuantileDMatrix。
    ref 用于验证集复用训练集分位点切分，减少额外构建开销并保持一致性。
    """
    it = CSRChunkDataIter(
        chunk_x_files=chunk_x_files,
        chunk_y_files=chunk_y_files,
        row_begin=row_begin,
        row_end=row_end,
        name=name,
        row_indices=row_indices,
    )
    print("[ITER-%s] build QuantileDMatrix rows=[%d, %d)" % (name, row_begin, row_end))
    try:
        return xgb.QuantileDMatrix(it, max_bin=int(max_bin), ref=ref)
    except TypeError:
        # 兼容不支持 max_bin/ref 关键字的旧版 XGBoost。
        if ref is not None:
            try:
                return xgb.QuantileDMatrix(it, ref=ref)
            except TypeError:
                pass
        return xgb.QuantileDMatrix(it)



def split_name_for_global_index(idx, split_ranges):
    """Return train/val/test according to global row index."""
    idx = int(idx)
    for name, (begin, end) in split_ranges.items():
        if idx >= int(begin) and idx < int(end):
            return str(name)
    return "unknown"


def export_sample_prediction_csv(model, chunk_x_files, chunk_y_files, chunk_meta_files,
                                 split_ranges, out_csv, split_assignment=None):
    """
    After training, reload each cached X chunk, predict each sample, and save one
    sample-level CSV for independent evaluation.

    CSV columns:
        true_label,pred_label,file_idx,cycle_index,split_type

    Extra columns are also saved for debugging:
        global_index,label_index

    The function predicts chunk-by-chunk, so it does not vstack all X chunks.
    """
    ensure_dir(os.path.dirname(out_csv))

    if len(chunk_x_files) != len(chunk_y_files):
        raise ValueError("chunk_x_files count != chunk_y_files count")
    if chunk_meta_files is not None and len(chunk_meta_files) != len(chunk_x_files):
        raise ValueError("chunk_meta_files count != chunk_x_files count")

    rows_written = 0
    global_offset = 0
    first = True

    for chunk_id, (xf, yf) in enumerate(zip(chunk_x_files, chunk_y_files)):
        t0 = time.time()
        X = load_npz(xf).tocsr()
        y = np.load(yf).reshape(-1)
        pred = model.predict(xgb.DMatrix(X)).reshape(-1)

        n = int(y.shape[0])
        if pred.shape[0] != n:
            raise RuntimeError("prediction rows mismatch at chunk %d" % chunk_id)

        if chunk_meta_files is not None:
            meta = np.load(chunk_meta_files[chunk_id])
            if meta.shape[0] != n:
                raise RuntimeError("meta rows mismatch at chunk %d" % chunk_id)
            file_idx = meta[:, 0].astype(np.int64)
            label_idx = meta[:, 1].astype(np.int64)
            cycle_index = meta[:, 2].astype(np.int64)
        else:
            # Fallback for old cache without metadata. Jump evaluation will be less reliable.
            file_idx = np.zeros(n, dtype=np.int64)
            label_idx = np.arange(global_offset, global_offset + n, dtype=np.int64)
            cycle_index = label_idx.copy()

        global_index = np.arange(global_offset, global_offset + n, dtype=np.int64)
        if split_assignment is not None:
            split_type = np.asarray(split_assignment[global_index], dtype=object)
        else:
            split_type = np.asarray([
                split_name_for_global_index(i, split_ranges) for i in global_index
            ], dtype=object)

        df = pd.DataFrame({
            "true_label": y.astype(np.float64),
            "pred_label": pred.astype(np.float64),
            "file_idx": file_idx,
            "cycle_index": cycle_index,
            "split_type": split_type,
            "global_index": global_index,
            "label_index": label_idx,
        })

        df.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        rows_written += n
        global_offset += n

        print("[PRED-CSV] chunk %d/%d rows=%d saved_rows=%d time=%.2fs" % (
            chunk_id + 1, len(chunk_x_files), n, rows_written, time.time() - t0
        ))

        del X, y, pred, df

    print("[SAVE] prediction csv:", out_csv, "rows=", rows_written)
    return out_csv

def load_chunks_as_csr(chunk_x_files, chunk_y_files):
    """
    保留旧函数，避免外部代码依赖时报错。
    主训练流程已不再调用该函数。
    """
    Xs = []
    ys = []

    for i, (xf, yf) in enumerate(zip(chunk_x_files, chunk_y_files)):
        t0 = time.time()
        X = load_npz(xf).tocsr()
        y = np.load(yf).reshape(-1, 1)
        Xs.append(X)
        ys.append(y)
        print("[LOAD] chunk %d/%d X=%s nnz=%d y=%s time=%.2fs" % (
            i + 1, len(chunk_x_files), str(X.shape), X.nnz, str(y.shape), time.time() - t0
        ))

    print("[LOAD] vstack chunks...")
    t0 = time.time()
    X_all = vstack(Xs, format="csr")
    y_all = np.vstack(ys).reshape(-1, 1)

    print("[LOAD] all X=%s nnz=%d y=%s time=%.2fs" % (
        str(X_all.shape), X_all.nnz, str(y_all.shape), time.time() - t0
    ))

    del Xs, ys
    return X_all, y_all


# -----------------------------
# Feature naming without external fmap
# -----------------------------

def decode_feature_id(fid, raw_ncols, seq_len, before_cycles):
    """
    将 XGBoost 的 f{fid} 映射成：
      raw_col, bc, type

    layout:
      [level offset0..seq_len-1][tog offset0..seq_len-1]

    offset=0 对应最早的窗口点：
      bc = before_cycles + seq_len - 1 - offset

    例如 seq_len=8, before_cycles=6:
      offset=0 -> bc13
      offset=7 -> bc6
    """
    raw_ncols = int(raw_ncols)
    seq_len = int(seq_len)
    type_span = raw_ncols * seq_len

    if int(fid) < type_span:
        feat_type = "level"
        local = int(fid)
    else:
        feat_type = "tog"
        local = int(fid) - type_span

    offset = int(local // raw_ncols)
    raw_col = int(local % raw_ncols)
    bc = int(before_cycles) + seq_len - 1 - offset
    return raw_col, bc, feat_type


def feature_name_from_id(fid, raw_ncols, seq_len, before_cycles):
    raw_col, bc, feat_type = decode_feature_id(fid, raw_ncols, seq_len, before_cycles)
    return "%d_bc%d_%s" % (raw_col, bc, feat_type)


def used_feature_ids_from_model(model):
    """
    获取模型中实际用到的 feature id。
    优先使用 get_score，必要时从 dump 中解析 f123。
    """
    used = set()

    for importance_type in ["gain", "weight", "cover"]:
        try:
            score = model.get_score(importance_type=importance_type)
            for k in score.keys():
                if isinstance(k, str) and k.startswith("f"):
                    used.add(int(k[1:]))
        except Exception:
            pass

    if used:
        return sorted(used)

    # fallback: parse text dump
    try:
        dumps = model.get_dump(with_stats=True)
        pat = re.compile(r"\[f(\d+)<")
        for tree in dumps:
            for m in pat.finditer(tree):
                used.add(int(m.group(1)))
    except Exception:
        pass

    return sorted(used)


def save_used_feature_map(model, fmap_file, raw_ncols, seq_len, before_cycles, feature_names=None):
    """
    只为模型实际用到的特征生成 fmap，避免生成 5600w 行 fmap。
    XGBoost 通常可以接受只包含 used features 的 fmap。
    如果你的 xgboost 版本不兼容 partial fmap，请使用 --no-dump-model。
    """
    used_ids = used_feature_ids_from_model(model)
    ensure_dir(os.path.dirname(fmap_file))

    with open(fmap_file, "w") as f:
        fmap_ids = range(len(feature_names)) if feature_names is not None else used_ids
        for fid in fmap_ids:
            if feature_names is not None and fid < len(feature_names):
                name = str(feature_names[fid]).replace("[", "_").replace("]", "_").replace("<", "_")
            else:
                name = feature_name_from_id(fid, raw_ncols, seq_len, before_cycles)
            # fmap format: <feature_id> <feature_name> <feature_type>
            f.write("%d\t%s\tq\n" % (fid, name))

    return used_ids


# -----------------------------
# Model
# -----------------------------

class XGBPowerPredictor(object):
    def __init__(self, num_rounds):
        self.num_rounds = int(num_rounds)
        self.xgb_model = None
        self.pred = None

    def load_train_val_data(self, X, y, fold=8):
        fold = int(fold)
        start = int(X.shape[0] * (fold - 2) / fold)
        end = int(X.shape[0] * (fold - 1) / fold)

        self.train_X = X[:start, :]
        self.train_y = y[:start, :].reshape(-1, 1)

        self.val_X = X[start:end, :]
        self.val_y = y[start:end, :].reshape(-1, 1)

        print("[TRAIN] train X=%s y=%s" % (str(self.train_X.shape), str(self.train_y.shape)))
        print("[TRAIN] val   X=%s y=%s" % (str(self.val_X.shape), str(self.val_y.shape)))

        self.dtrain = xgb.DMatrix(self.train_X, label=self.train_y)
        self.dval = xgb.DMatrix(self.val_X, label=self.val_y)

    def load_train_val_data_iter(self, chunk_x_files, chunk_y_files, fold=8, max_bin=32,
                                 shuffle_seed=108, chunk_meta_files=None,
                                 fixed_test_file_idx=None, fixed_test_file_indices=None):
        fold = int(fold)
        chunk_rows, total_rows = count_chunk_rows(chunk_y_files)

        rng = np.random.RandomState(int(shuffle_seed))

        if fixed_test_file_idx is None and fixed_test_file_indices is None:
            start = int(total_rows * (fold - 2) / fold)
            end = int(total_rows * (fold - 1) / fold)
            shuffle_idx = rng.permutation(int(total_rows)).astype(np.int64, copy=False)

            train_indices = shuffle_idx[:start]
            val_indices = shuffle_idx[start:end]
            test_indices = shuffle_idx[end:]
            split_mode = "random-all-samples"
        else:
            if chunk_meta_files is None:
                raise ValueError("fixed file split requires chunk metadata")

            file_idx_parts = []
            for meta_file in chunk_meta_files:
                meta = np.load(meta_file, mmap_mode="r")
                file_idx_parts.append(np.asarray(meta[:, 0], dtype=np.int64))
            sample_file_idx = np.concatenate(file_idx_parts)
            if sample_file_idx.shape[0] != int(total_rows):
                raise RuntimeError("chunk metadata row count != total rows")

            if fixed_test_file_indices is None:
                fixed_ids = np.asarray([int(fixed_test_file_idx)], dtype=np.int64)
            else:
                fixed_ids = np.asarray(fixed_test_file_indices, dtype=np.int64).reshape(-1)
            test_mask = np.isin(sample_file_idx, fixed_ids)
            test_indices = np.flatnonzero(test_mask).astype(np.int64, copy=False)
            train_pool_indices = np.flatnonzero(~test_mask).astype(np.int64, copy=False)
            if test_indices.size == 0:
                raise ValueError("fixed pipeline test file indices have no samples: %s" % fixed_ids.tolist())
            if train_pool_indices.size < fold:
                raise ValueError("training pool is too small for fold=%d" % fold)

            shuffled_train_pool = rng.permutation(train_pool_indices)
            val_rows = int(train_pool_indices.size / fold)
            val_indices = shuffled_train_pool[-val_rows:]
            train_indices = shuffled_train_pool[:-val_rows]
            start = int(train_indices.size)
            end = int(train_indices.size + val_indices.size)
            split_mode = "fixed-pipeline-test-files"

        split_assignment = np.empty(int(total_rows), dtype=object)
        split_assignment[train_indices] = "train"
        split_assignment[val_indices] = "val"
        split_assignment[test_indices] = "test"

        self.total_rows = int(total_rows)
        self.train_begin = 0
        self.train_end = int(start)
        self.val_begin = int(start)
        self.val_end = int(end)
        self.test_begin = int(end)
        self.test_end = int(total_rows)

        self.train_rows = int(start)
        self.val_rows = int(end - start)
        self.test_rows = int(total_rows - end)
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.split_assignment = split_assignment

        print("[TRAIN] total rows=%d chunks=%d" % (total_rows, len(chunk_rows)))
        print("[TRAIN] split mode: %s" % split_mode)
        if fixed_test_file_idx is not None or fixed_test_file_indices is not None:
            print("[TRAIN] fixed test file indices=%s" % fixed_ids.tolist())
        print("[TRAIN] shuffle training pool before val split: enabled seed=%d" % int(shuffle_seed))
        print("[TRAIN] train n=%d" % self.train_rows)
        print("[TRAIN] val   n=%d" % self.val_rows)
        print("[TRAIN] test  n=%d" % self.test_rows)

        self.dtrain = make_quantile_dmatrix_from_chunks(
            chunk_x_files=chunk_x_files,
            chunk_y_files=chunk_y_files,
            row_begin=0,
            row_end=total_rows,
            name="train",
            max_bin=max_bin,
            ref=None,
            row_indices=train_indices,
        )
        self.dval = make_quantile_dmatrix_from_chunks(
            chunk_x_files=chunk_x_files,
            chunk_y_files=chunk_y_files,
            row_begin=0,
            row_end=total_rows,
            name="val",
            max_bin=max_bin,
            ref=self.dtrain,
            row_indices=val_indices,
        )

    def train(self, params, resume=False, model_file=""):
        watchlist = [(self.dval, "eval"), (self.dtrain, "train")]

        if resume:
            print("[TRAIN] resume from", model_file)
            self.load_model(model_file)
            self.xgb_model = xgb.train(
                params=params,
                dtrain=self.dtrain,
                num_boost_round=self.num_rounds,
                evals=watchlist,
                verbose_eval=1,
                xgb_model=self.xgb_model,
            )
        else:
            self.xgb_model = xgb.train(
                params=params,
                dtrain=self.dtrain,
                num_boost_round=self.num_rounds,
                evals=watchlist,
                verbose_eval=1,
                early_stopping_rounds=30,
            )

    def load_model(self, model_file):
        self.xgb_model = xgb.Booster({"nthread": 32})
        self.xgb_model.load_model(model_file)

    def save_model(self, model_file, tree_file, raw_ncols, seq_len, before_cycles,
                   fmap_file=None, no_dump=False, feature_names=None):
        ensure_dir(os.path.dirname(model_file))
        self.xgb_model.save_model(model_file)
        print("[SAVE] model:", model_file)

        if no_dump:
            print("[SKIP] tree dump disabled")
            return

        if fmap_file is None:
            fmap_file = tree_file + ".used.fmap"

        used_ids = save_used_feature_map(
            model=self.xgb_model,
            fmap_file=fmap_file,
            raw_ncols=raw_ncols,
            seq_len=seq_len,
            before_cycles=before_cycles,
            feature_names=feature_names,
        )

        print("[SAVE] used feature map:", fmap_file, "used_features=", len(used_ids))

        try:
            self.xgb_model.dump_model(
                tree_file,
                fmap=fmap_file,
                with_stats=True,
                dump_format="json",
            )
            print("[SAVE] tree dump:", tree_file)
        except Exception as e:
            print("[WARN] dump_model with partial fmap failed:", str(e))
            print("[WARN] retry dump_model without fmap")
            self.xgb_model.dump_model(
                tree_file,
                with_stats=True,
                dump_format="json",
            )
            print("[SAVE] tree dump without fmap:", tree_file)

    def save_feature_importance(self, importance_text_file, indice_out_dir,
                                raw_ncols, seq_len, before_cycles, top_ns=None,
                                feature_names=None):
        if top_ns is None:
            top_ns = [512]

        ensure_dir(os.path.dirname(importance_text_file))
        ensure_dir(indice_out_dir)

        score = self.xgb_model.get_score(importance_type="gain")

        rows = []
        for feat, gain in score.items():
            if not isinstance(feat, str) or not feat.startswith("f"):
                continue
            fid = int(feat[1:])
            if feature_names is not None and fid < len(feature_names):
                raw_col, bc, feat_type = fid, "", "step2_proxy"
                signal_name = str(feature_names[fid])
            else:
                raw_col, bc, feat_type = decode_feature_id(
                    fid=fid, raw_ncols=raw_ncols, seq_len=seq_len,
                    before_cycles=before_cycles)
                signal_name = "%d_bc%d_%s" % (raw_col, bc, feat_type)
            rows.append({
                "xgb_feature": feat,
                "fid": fid,
                "raw_col": raw_col,
                "bc": bc,
                "type": feat_type,
                "signal_name": signal_name,
                "gain": gain,
            })

        columns = ["xgb_feature", "fid", "raw_col", "bc", "type", "signal_name", "gain"]
        if not rows:
            print("[WARN] no feature importance found")
            pd.DataFrame(columns=columns).to_csv(importance_text_file, index=False)
            return

        df = pd.DataFrame(rows).sort_values("gain", ascending=False)
        df.to_csv(importance_text_file, index=False)
        print("[SAVE] feature importance:", importance_text_file)

        for top_n in top_ns:
            top_n = int(top_n)
            top_df = df.head(top_n)

            raw_file = os.path.join(indice_out_dir, "%d_feature_raw_cols.txt" % top_n)
            name_file = os.path.join(indice_out_dir, "%d_feature_names.txt" % top_n)
            detail_file = os.path.join(indice_out_dir, "%d_feature_detail.csv" % top_n)

            # raw_col 可能在不同 bc/type 重复；这里输出去重 raw_col 和完整 detail。
            unique_raw_cols = top_df["raw_col"].drop_duplicates().astype(int).values

            with open(raw_file, "w") as f:
                for c in unique_raw_cols:
                    f.write("%d\n" % int(c))

            with open(name_file, "w") as f:
                for name in top_df["signal_name"].values:
                    f.write("%s\n" % str(name))

            top_df.to_csv(detail_file, index=False)

            print("[SAVE] top%d raw cols: %s unique=%d" % (top_n, raw_file, len(unique_raw_cols)))
            print("[SAVE] top%d names   : %s" % (top_n, name_file))
            print("[SAVE] top%d detail  : %s" % (top_n, detail_file))

    def plot_importance_from_csv(self, importance_csv, fig_file, max_num_features=100):
        df = pd.read_csv(importance_csv)
        if df.empty:
            print("[WARN] empty importance csv, skip plot")
            return

        df = df.head(int(max_num_features))
        plt.figure(figsize=(14, max(6, int(max_num_features) * 0.16)))
        plt.barh(np.arange(len(df)), df["gain"].values)
        plt.yticks(np.arange(len(df)), df["signal_name"].values, fontsize=6)
        plt.gca().invert_yaxis()
        plt.xlabel("gain")
        plt.tight_layout()
        ensure_dir(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, bbox_inches="tight")
        plt.close()
        print("[SAVE] importance figure:", fig_file)


# -----------------------------
# XGBoost params and output dirs
# -----------------------------

def setup_directories(hpotag, inst, before_cycles):
    base_dir = "./%s/%s-bc%s" % (hpotag, inst, str(before_cycles))
    dirs = {
        "indices": os.path.join(base_dir, "indices"),
        "result": os.path.join(base_dir, "result"),
        "pred": os.path.join(base_dir, "pred"),
        "eval": os.path.join(base_dir, "eval"),
    }
    for d in dirs.values():
        ensure_dir(d)
    return dirs


def build_params(args):
    params = {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "tree_method": args.tree_method,
        "max_bin": int(args.max_bin),
        "grow_policy": "lossguide",
        "max_depth": int(args.max_depth),
        "max_leaves": int(args.max_leaves),
        "learning_rate": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "colsample_bynode": float(args.colsample_bynode),
        "nthread": int(args.nthread),
        "seed": int(args.seed),
        "eval_metric": "rmse",
    }

    if args.device:
        params["device"] = args.device

    return params


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-Step2 XGBoost validation using selected proxies from every combo"
    )

    parser.add_argument("--tw-manifest", required=True, help="Step1 timewindow_manifest.json")
    parser.add_argument("--proxy-manifest", required=True, help="Step2 mcp_per_offset_manifest.json")
    parser.add_argument("--namefile", default="", help="raw signal namefile for proxy traceability")
    parser.add_argument(
        "--combos", default="",
        help="comma-separated combo keys to use, in feature concatenation order; empty uses every Step1 combo")

    parser.add_argument("--rounds", "-r", type=int, default=200, help="XGBoost boosting rounds")
    parser.add_argument("--length", "-l", type=int, default=8, help="seq_len")
    parser.add_argument("--before-cycles", "-bc", type=int, default=6, help="before_cycles")
    parser.add_argument("--label-start-cycle", type=int, default=20, help="cycle of label[0]; v2_3 no longer drops the first raw label")
    parser.add_argument("--label-scale", type=float, default=1000000.0, help="scale label to W; default assumes label is uW")

    parser.add_argument("--cache-dir", type=str, default="./cache_ultrawide_no_topk", help="dataset cache directory")
    parser.add_argument(
        "--build-block-size", type=int, default=200,
        help="rows per combo-proxy cache chunk; smaller values bound combo-stitching memory at the cost of more disk I/O; <=0 uses one chunk per case")
    parser.add_argument("--data-dtype", choices=["float32", "float64"], default="float32", help="feature data dtype in generated cache")
    parser.add_argument("--force-rebuild-dataset", action="store_true", help="kept for legacy CLI compatibility")
    parser.add_argument("--build-dataset-only", action="store_true", help="build combo-proxy chunks and stop before training")

    parser.add_argument("--target", "-t", type=str, default="tile_reset_domain", help="prediction target name")
    parser.add_argument("--hpotag", "-hpo", type=str, default="allbm_iso", help="output experiment tag")
    parser.add_argument("--fold", type=int, default=8, help="train/val split fold, val uses [fold-2, fold-1)")
    parser.add_argument("--shuffle-seed", type=int, default=108, help="random seed for shuffling before train/val/test split")
    parser.add_argument("--fixed-test-last-file", action="store_true", help="use the last input file only as test; split validation only from the preceding training files")
    parser.add_argument("--top-ns", type=str, default="4096", help="topN list, e.g. '512,4096'")
    parser.add_argument("--no-dump-model", action="store_true", help="skip json tree dump to reduce output time/size")
    parser.add_argument("--plot-importance", action="store_true", help="plot importance figure")

    # XGBoost params tuned for no-TopK ultra-wide sparse features
    parser.add_argument("--tree-method", choices=["hist", "approx"], default="hist")
    parser.add_argument("--max-bin", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-leaves", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.01)
    parser.add_argument("--colsample-bynode", type=float, default=0.01)
    parser.add_argument("--nthread", type=int, default=32)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--device", type=str, default="", help="optional: cuda for newer xgboost")

    args = parser.parse_args()
    selected_combos = [key.strip() for key in args.combos.split(",") if key.strip()]

    dirs = setup_directories(args.hpotag, args.target, args.before_cycles)

    model_f = os.path.join(dirs["result"], "%s-xgbmodel-%d.json" % (args.target, args.rounds))
    tree_f = os.path.join(dirs["result"], "%s-xgbtree-%d.json" % (args.target, args.rounds))
    fmap_used_f = os.path.join(dirs["result"], "%s-used-feature-map.fmap" % args.target)
    importance_csv = os.path.join(dirs["result"], "%s-feature_importance_%d.csv" % (args.target, args.rounds))
    importance_fig = os.path.join(dirs["result"], "%s-feature_importance_%d.png" % (args.target, args.rounds))

    print("========== PIPELINE VALIDATION CONFIG ==========")
    print("target           :", args.target)
    print("step1 manifest   :", args.tw_manifest)
    print("step2 manifest   :", args.proxy_manifest)
    print("selected combos  :", selected_combos if selected_combos else "all Step1 combos")
    print("cache_dir        :", args.cache_dir)
    print("build block rows :", args.build_block_size)
    print("legacy rounds    :", args.rounds)
    print("legacy XGB params: unchanged")
    print("================================================")

    chunk_x_files, chunk_y_files, pipeline_meta = build_pipeline_chunks(
        tw_manifest=args.tw_manifest,
        proxy_manifest=args.proxy_manifest,
        cache_dir=args.cache_dir,
        label_scale=args.label_scale,
        data_dtype=args.data_dtype,
        namefile=args.namefile or None,
        block_size=args.build_block_size,
        combo_keys=selected_combos,
    )
    raw_ncols = int(pipeline_meta["total_columns"])
    feature_names = list(pipeline_meta["feature_names"])
    print("[PIPELINE-DATA] combos=%d columns=%d rows=%d train_files=%d test_files=%d" % (
        len(pipeline_meta["combo_order"]), raw_ncols, int(pipeline_meta["total_rows"]),
        len(pipeline_meta["train_file_indices"]), len(pipeline_meta["test_file_indices"])))
    if args.build_dataset_only:
        print("[DONE] dataset-only")
        return

    params = build_params(args)
    print("[XGB PARAMS]", params)

    chunk_meta_files = list_chunk_meta_files(chunk_x_files)
    if chunk_meta_files is None:
        print("[WARN] chunk meta files not found; file_idx/cycle_index will fall back to global index")

    predictor = XGBPowerPredictor(args.rounds)
    predictor.load_train_val_data_iter(
        chunk_x_files=chunk_x_files,
        chunk_y_files=chunk_y_files,
        fold=args.fold,
        max_bin=args.max_bin,
        shuffle_seed=args.shuffle_seed,
        chunk_meta_files=chunk_meta_files,
        fixed_test_file_indices=pipeline_meta["test_file_indices"],
    )
    predictor.train(params)

    split_ranges = {
        "train": (predictor.train_begin, predictor.train_end),
        "val": (predictor.val_begin, predictor.val_end),
        "test": (predictor.test_begin, predictor.test_end),
    }
    prediction_csv = os.path.join(
        dirs["eval"],
        "%s_prediction_samples_%d.csv" % (args.target, args.rounds)
    )
    export_sample_prediction_csv(
        model=predictor.xgb_model,
        chunk_x_files=chunk_x_files,
        chunk_y_files=chunk_y_files,
        chunk_meta_files=chunk_meta_files,
        split_ranges=split_ranges,
        out_csv=prediction_csv,
        split_assignment=predictor.split_assignment,
    )

    evaluate_prediction_csv(
        prediction_csv=prediction_csv,
        out_dir=dirs["eval"],
        require_consecutive=True,
        threshold_ratios=[x / 10.0 for x in range(1, 10)],
        plot_jump_hist=True,
        jump_hist_bin_step=10.0,
        plot_per_file=False,
    )

    top_ns = parse_top_ns(args.top_ns)
    predictor.save_feature_importance(
        importance_text_file=importance_csv,
        indice_out_dir=dirs["indices"],
        raw_ncols=raw_ncols,
        seq_len=args.length,
        before_cycles=args.before_cycles,
        top_ns=top_ns,
        feature_names=feature_names,
    )

    if args.plot_importance:
        predictor.plot_importance_from_csv(
            importance_csv=importance_csv,
            fig_file=importance_fig,
            max_num_features=min(100, max(top_ns)),
        )

    predictor.save_model(
        model_file=model_f,
        tree_file=tree_f,
        raw_ncols=raw_ncols,
        seq_len=args.length,
        before_cycles=args.before_cycles,
        fmap_file=fmap_used_f,
        no_dump=args.no_dump_model,
        feature_names=feature_names,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
