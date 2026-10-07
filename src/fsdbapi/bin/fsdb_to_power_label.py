#!/usr/bin/env python3
"""Extract cycle-aligned power labels from a power FSDB.

For each clock-edge sample time, the exporter selects the nearest discrete power
value-change point in the FSDB and writes both an audit TSV and a training .npy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def find_tool(fsdbapi_root: Path) -> Path:
    return fsdbapi_root / "build" / "fsdb_to_power_label"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_power_column(tsv_path: Path, drop_first_sample: bool, dtype: str) -> np.ndarray:
    values: list[float] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            values.append(float(row["power"]))
    if drop_first_sample and values:
        values = values[1:]
    return np.asarray(values, dtype=dtype).reshape(-1, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdb", required=True, help="输入功耗 FSDB")
    ap.add_argument("--power-signal", default="", help="功耗信号完整名或唯一后缀")
    ap.add_argument("--output-dir", default="", help="输出目录")
    ap.add_argument("--dataset-name", default="", help="输出文件名前缀，默认取 FSDB stem")
    ap.add_argument("--step", default="", help="周期上升沿采样间隔，单位为 FSDB tag")
    ap.add_argument("--start", default="", help="周期上升沿起点，默认使用 FSDB min tag")
    ap.add_argument("--end", default="", help="周期上升沿终点，默认使用 FSDB max tag")
    ap.add_argument("--end-inclusive", action="store_true", help="包含 end 采样点")
    ap.add_argument("--drop-first-sample", action="store_true", help="丢弃第一个功耗样本，用于对齐 drop-first TOG")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="npy 保存精度")
    ap.add_argument("--fsdbapi-root", default="", help="fsdbapi 根目录，默认从本脚本位置推导")
    ap.add_argument("--list-signals", default="", help="只列出可作为功耗的 numeric 信号候选")
    ap.add_argument("--auto-build", action="store_true", help="缺少二进制时自动 make")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    script = Path(__file__).resolve()
    fsdbapi_root = Path(args.fsdbapi_root).expanduser().resolve() if args.fsdbapi_root else script.parents[1]
    tool = find_tool(fsdbapi_root)
    if args.auto_build or not tool.exists():
        run(["make", "-C", str(fsdbapi_root), "fsdb_to_power_label"])
    if not tool.exists():
        raise SystemExit(f"missing binary: {tool}")

    fsdb = Path(args.fsdb).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(".").resolve()
    if args.output_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or fsdb.stem

    if args.list_signals:
        cmd = [str(tool), "--fsdb", str(fsdb), "--list-signals", args.list_signals]
        run(cmd)
        return 0

    if not args.power_signal:
        raise SystemExit("--power-signal is required unless --list-signals is used")
    if not args.output_dir:
        raise SystemExit("--output-dir is required unless --list-signals is used")
    if not args.step:
        raise SystemExit("--step is required unless --list-signals is used")

    tsv_path = out_dir / f"{dataset_name}.power.samples.tsv"
    meta_path = out_dir / f"{dataset_name}.power.meta.json"
    npy_path = out_dir / f"{dataset_name}.npy"

    cmd = [
        str(tool),
        "--fsdb", str(fsdb),
        "--power-signal", args.power_signal,
        "--output-tsv", str(tsv_path),
        "--output-meta", str(meta_path),
        "--step", str(args.step),
    ]
    if args.start:
        cmd += ["--start", str(args.start)]
    if args.end:
        cmd += ["--end", str(args.end)]
    if args.end_inclusive:
        cmd.append("--end-inclusive")
    if args.verbose:
        cmd.append("--verbose")
    run(cmd)

    arr = read_power_column(tsv_path, args.drop_first_sample, args.dtype)
    np.save(npy_path, arr)

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({
        "output_npy": str(npy_path),
        "npy_shape": list(arr.shape),
        "npy_dtype": str(arr.dtype),
        "drop_first_sample": bool(args.drop_first_sample),
        "sampler_rule": "nearest discrete power value to each clock-edge sample; ties choose earlier event",
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"power npy: {npy_path}")
    print(f"shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"audit tsv: {tsv_path}")
    print(f"meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
