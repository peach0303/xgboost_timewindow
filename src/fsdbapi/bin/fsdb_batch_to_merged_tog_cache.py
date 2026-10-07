#!/usr/bin/env python3
"""Export multiple FSDB/CSV files into one merged TOG cache for global reduction."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "bin" / "fsdb_to_tog_cache.py"
MERGER = ROOT / "bin" / "merge_tog_cache_manifests.py"
CSV_CACHE_BUILDER_CANDIDATES = [
    ROOT.parent / "steps" / "step4-dataset-organization" / "dimentionality_reduction_fromcache.py",
    ROOT.parent / "powermodeling-demo" / "src" / "step4-dataset-organization" / "dimentionality_reduction_fromcache.py",
    ROOT.parent / "hw-fsdb-npz" / "steps" / "step4-dataset-organization" / "dimentionality_reduction_fromcache.py",
    ROOT.parent.parent / "powermodeling-demo" / "src" / "step4-dataset-organization" / "dimentionality_reduction_fromcache.py",
    ROOT.parent.parent / "hw-fsdb-npz" / "steps" / "step4-dataset-organization" / "dimentionality_reduction_fromcache.py",
]


def find_csv_cache_builder() -> Path:
    for candidate in CSV_CACHE_BUILDER_CANDIDATES:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(p) for p in CSV_CACHE_BUILDER_CANDIDATES)
    raise SystemExit(f"missing CSV cache builder; searched:\n  {searched}")


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def safe_dataset_name(path: Path, used: set[str]) -> str:
    base = path.stem.replace("/", "_")
    name = base
    i = 1
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def read_paths(items: list[str] | None, list_path: str | None, label: str) -> list[Path]:
    paths = [Path(x).resolve() for x in (items or [])]
    if list_path:
        with Path(list_path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                s = line.strip()
                if s and not s.startswith("#"):
                    paths.append(Path(s).resolve())
    if not paths:
        raise SystemExit(f"no {label} inputs; use --{label} or --{label}-list")
    return paths


def read_fsdbs(args: argparse.Namespace) -> list[Path]:
    return read_paths(args.fsdb, args.fsdb_list, "fsdb")


def read_csvs(args: argparse.Namespace) -> list[Path]:
    return read_paths(args.csv, args.csv_list, "csv")


def write_filelist(paths: list[Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in paths:
            f.write(str(item) + "\n")


def export_csvs(args: argparse.Namespace) -> None:
    csvs = read_csvs(args)
    output_dir = Path(args.output_dir).resolve()
    work_dir = output_dir / "_work"
    filelist = work_dir / "csv_input_filelist.txt"
    write_filelist(csvs, filelist)

    csv_cache_builder = find_csv_cache_builder()

    cmd = [
        sys.executable,
        str(csv_cache_builder),
        "--filelist",
        str(filelist),
        "--output-dir",
        str(output_dir),
        "--output-meta",
        str(work_dir / "csv_input_cache_meta.json"),
        "--output-cols",
        str(work_dir / "csv_input_cache.cols.txt"),
        "--output-original-headers",
        str(output_dir / "original_headers.txt"),
        "--max-mem",
        args.max_mem,
        "--row-chunksize",
        str(args.row_chunksize),
        "--raw-col-block",
        str(args.raw_col_block),
        "--cache-format",
        "csc",
        "--keep-cache",
        "--disable-zero-remove",
        "--disable-one-merge",
        "--disable-duplicate-remove",
        "--disable-sim-reduce",
    ]
    run(cmd)
    print(f"CSV cache manifest: {output_dir / '_work' / 'tog_cache_manifest.json'}", flush=True)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch FSDB/CSV -> merged TOG cache for global dimensionality reduction.")
    p.add_argument("--input-format", choices=["fsdb", "csv"], default="fsdb",
                   help="Input data format. csv means CSV files already contain TOG data with time in the first column.")
    p.add_argument("--fsdb", action="append", help="Input FSDB. Repeatable.")
    p.add_argument("--fsdb-list", help="Text file with one FSDB path per line.")
    p.add_argument("--csv", action="append", help="Input TOG CSV. Repeatable.")
    p.add_argument("--csv-list", help="Text file with one TOG CSV path per line.")
    p.add_argument("--output-dir", required=True, help="Final merged output directory.")
    p.add_argument("--start", type=int, help="First FSDB sample tag. Required for --input-format fsdb.")
    p.add_argument("--end", type=int, help="End FSDB sample tag. Required for --input-format fsdb.")
    p.add_argument("--step", type=int, default=1, help="FSDB sample step.")
    p.add_argument("--signal-list", help="Optional signal filter file for FSDB input.")
    p.add_argument("--raw-col-block", type=int, default=20000)
    p.add_argument("--row-chunksize", type=int, default=1024, help="CSV reader row chunk size.")
    p.add_argument("--max-mem", default="28GB", help="CSV cache builder memory budget.")
    p.add_argument("--tog-mode", choices=["sample-xor", "any-change"], default="sample-xor")
    p.add_argument("--end-inclusive", action="store_true")
    p.add_argument("--drop-first-tog-row", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-level-cache", action="store_true", help="Do not write diff-before level cache.")
    return p.parse_args()


def export_fsdbs(args: argparse.Namespace) -> None:
    fsdbs = read_fsdbs(args)
    output_dir = Path(args.output_dir).resolve()
    source_root = output_dir / "_source_exports"
    source_root.mkdir(parents=True, exist_ok=True)

    manifests: list[Path] = []
    used_names: set[str] = set()
    for fsdb in fsdbs:
        dataset = safe_dataset_name(fsdb, used_names)
        per_out = source_root / dataset
        cmd = [
            sys.executable,
            str(EXPORTER),
            "--fsdb",
            str(fsdb),
            "--output-dir",
            str(per_out),
            "--dataset-name",
            dataset,
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--step",
            str(args.step),
            "--raw-col-block",
            str(args.raw_col_block),
            "--tog-mode",
            args.tog_mode,
        ]
        if args.end_inclusive:
            cmd += ["--end-inclusive"]
        if args.drop_first_tog_row:
            cmd += ["--drop-first-tog-row"]
        if args.signal_list:
            cmd += ["--signal-list", str(Path(args.signal_list).resolve())]
        if args.no_level_cache:
            cmd += ["--no-level-cache"]
        if args.rebuild:
            cmd += ["--rebuild"]
            args.rebuild = False
        else:
            cmd += ["--no-build"]
        if args.verbose:
            cmd += ["--verbose"]
        run(cmd)
        manifests.append(per_out / "_work" / "tog_cache_manifest.json")

    merge_cmd = [sys.executable, str(MERGER), "--output-dir", str(output_dir)]
    for manifest in manifests:
        merge_cmd += ["--manifest", str(manifest)]
    run(merge_cmd)


def main() -> int:
    args = parse_args()
    if args.input_format == "csv":
        export_csvs(args)
        return 0

    if args.start is None or args.end is None:
        raise SystemExit("--start and --end are required for --input-format fsdb")
    export_fsdbs(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
