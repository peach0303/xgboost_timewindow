#!/usr/bin/env python3
"""Build and run the FSDB -> bit-level TOG sparse-cache exporter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "fsdb_to_tog_cache"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def build_if_needed(force: bool) -> None:
    src = ROOT / "src" / "fsdb_to_tog_cache.cpp"
    if force or not BIN.exists() or BIN.stat().st_mtime < src.stat().st_mtime:
        run(["make"], cwd=ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export bus-level FSDB into X/Z-cleaned level and TOG CSC caches."
    )
    p.add_argument("--fsdb", required=True, help="Input FSDB file.")
    p.add_argument("--output-dir", required=True, help="Output reduction-style directory.")
    p.add_argument("--start", type=int, help="First sample FSDB tag, inclusive. Default: FSDB min simulation tag.")
    p.add_argument("--end", type=int, help="End sample FSDB tag, exclusive unless --end-inclusive is used. Default: FSDB max simulation tag.")
    p.add_argument("--step", required=True, type=int, help="Sample step in FSDB tags.")
    p.add_argument("--dataset-name", help="Cache filename stem. Default: FSDB filename stem.")
    p.add_argument("--signal-list", help="Optional signal filter file.")
    p.add_argument("--raw-col-block", type=int, default=20000, help="Raw bus signals per cache block.")
    p.add_argument(
        "--tog-mode",
        choices=["sample-xor", "any-change"],
        default="sample-xor",
        help="sample-xor matches level-to-TOG row XOR; any-change marks a bit if it changes within the interval.",
    )
    p.add_argument("--end-inclusive", action="store_true", help="Include --end as a sample point.")
    p.add_argument("--drop-first-tog-row", action="store_true", help="Drop the first dummy TOG row after diff.")
    p.add_argument("--list-signals", help="Write supported signal list and exit.")
    p.add_argument("--no-level-cache", action="store_true", help="Do not write the initial level row cache. Full level snapshots require initial level + TOG restoration.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild of the C++ exporter.")
    p.add_argument("--no-build", action="store_true", help="Do not build automatically.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_build:
        build_if_needed(args.rebuild)
    if not BIN.exists():
        raise SystemExit(f"missing exporter binary: {BIN}; run `make` in {ROOT}")

    cmd = [
        str(BIN),
        "--fsdb",
        str(Path(args.fsdb).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--step",
        str(args.step),
        "--raw-col-block",
        str(args.raw_col_block),
        "--tog-mode",
        args.tog_mode,
    ]
    if args.start is not None:
        cmd += ["--start", str(args.start)]
    if args.end is not None:
        cmd += ["--end", str(args.end)]
    if args.dataset_name:
        cmd += ["--dataset-name", args.dataset_name]
    if args.end_inclusive:
        cmd += ["--end-inclusive"]
    if args.drop_first_tog_row:
        cmd += ["--drop-first-tog-row"]
    if args.signal_list:
        cmd += ["--signal-list", str(Path(args.signal_list).resolve())]
    if args.list_signals:
        cmd += ["--list-signals", str(Path(args.list_signals).resolve())]
    if args.no_level_cache:
        cmd += ["--no-level-cache"]
    if args.verbose:
        cmd += ["--verbose"]

    run(cmd)

    if not args.list_signals:
        manifest = Path(args.output_dir).resolve() / "_work" / "tog_cache_manifest.json"
        with manifest.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        rows = meta.get("output_rows")
        if rows is None and meta.get("blocks"):
            first_rows = meta.get("blocks", [{}])[0].get("n_rows", {})
            rows = next(iter(first_rows.values()), "unknown")
        if rows is None:
            rows = "unknown"
        print(
            f"manifest: {manifest}\n"
            f"blocks: {meta.get('num_blocks')}  "
            f"xz_removed: {meta.get('xz_removed_count')}  "
            f"rows: {rows}  "
            f"level_cache: {meta.get('level_cache_available', False)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
