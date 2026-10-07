#!/usr/bin/env python3
"""Augment a target-window baseline with signals selected in another window.

Only signal identities cross the window boundary.  Feature values are always
rebuilt from the target Step1 manifest (for example offsets 12..6 or 12..3),
so no offset 5..0 value is accidentally fed to the target experiment.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xgboost as xgb

from step4_screen_features import build_top_matrices_from_step1


FEATURE_RE = re.compile(
    r"^(?P<data_type>tog|level)_o(?P<offset>-?\d+)_(?P<signal>.+)$"
)


def read_feature_tokens(path: str) -> list[str]:
    values = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if token and token.lower() not in {"proxy", "feature"}:
                values.append(token)
    return values


def signal_name(feature_name: str) -> str | None:
    match = FEATURE_RE.fullmatch(feature_name)
    return match.group("signal") if match else None


def source_feature(feature_name: str) -> tuple[int, str] | None:
    match = FEATURE_RE.fullmatch(feature_name)
    if not match:
        return None
    return int(match.group("offset")), match.group("signal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map source-window proxy signals onto target-window features"
    )
    parser.add_argument("--target-tw-manifest", required=True)
    parser.add_argument("--namefile", required=True)
    parser.add_argument(
        "--baseline-feature-names", required=True,
        help="Existing target-window feature_names.txt to reuse",
    )
    parser.add_argument(
        "--source-proxy-features", required=True,
        help="Feature/ranking file selected from the source window (for example 5..0)",
    )
    parser.add_argument("--source-top-signals", type=int, default=0)
    parser.add_argument("--source-offset-min", type=int, default=None)
    parser.add_argument("--source-offset-max", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.source_top_signals < 0:
        parser.error("--source-top-signals must be non-negative")
    if (
        args.source_offset_min is not None
        and args.source_offset_max is not None
        and args.source_offset_min > args.source_offset_max
    ):
        parser.error("--source-offset-min must not exceed --source-offset-max")

    output = Path(args.output_dir).resolve()
    metadata_path = output / "cross_window_union_meta.json"
    if metadata_path.exists() and not args.force:
        print(f"reuse existing cross-window features: {metadata_path}")
        return

    baseline_names = read_feature_tokens(args.baseline_feature_names)
    source_names = read_feature_tokens(args.source_proxy_features)
    parsed_source = [value for value in map(source_feature, source_names) if value]
    source_in_range = [
        (offset, signal) for offset, signal in parsed_source
        if (args.source_offset_min is None or offset >= args.source_offset_min)
        and (args.source_offset_max is None or offset <= args.source_offset_max)
    ]
    transferred_signals = list(OrderedDict.fromkeys(
        signal for _offset, signal in source_in_range
    ))
    source_unique_signals_in_range = len(transferred_signals)
    if args.source_top_signals > 0:
        transferred_signals = transferred_signals[:args.source_top_signals]
    if not baseline_names:
        raise SystemExit("ERROR: baseline feature list is empty")
    if not transferred_signals:
        raise SystemExit(
            "ERROR: no TYPE_oOFFSET_signal names remain after source offset filtering"
        )

    baseline_signals = {
        value for value in (signal_name(name) for name in baseline_names) if value
    }
    with open(args.target_tw_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=OrderedDict)
    all_signals = baseline_signals | set(transferred_signals)
    train_x, train_y, train_weight, test_x, test_y, test_weight, names, blocks = (
        build_top_matrices_from_step1(manifest, args.namefile, all_signals)
    )
    name_to_column = {name: index for index, name in enumerate(names)}
    missing_baseline = [name for name in baseline_names if name not in name_to_column]
    if missing_baseline:
        preview = ", ".join(missing_baseline[:5])
        raise SystemExit(
            f"ERROR: {len(missing_baseline)} baseline features are absent from target manifest: {preview}"
        )
    transfer_names = [
        name for name in names
        if signal_name(name) in set(transferred_signals) and name not in set(baseline_names)
    ]
    final_names = baseline_names + transfer_names
    columns = np.asarray([name_to_column[name] for name in final_names], dtype=np.int64)
    train_x = train_x[:, columns].tocsr()
    test_x = test_x[:, columns].tocsr()

    output.mkdir(parents=True, exist_ok=True)
    train_buffer = output / "train_cross_window.buffer"
    test_buffer = output / "test_cross_window.buffer"
    feature_names_file = output / "feature_names.txt"
    xgb.DMatrix(
        train_x, label=train_y, weight=train_weight, feature_names=final_names
    ).save_binary(train_buffer)
    xgb.DMatrix(
        test_x, label=test_y, weight=test_weight, feature_names=final_names
    ).save_binary(test_buffer)
    feature_names_file.write_text("\n".join(final_names) + "\n", encoding="utf-8")
    metadata = OrderedDict([
        ("schema_version", 1),
        ("experiment_tag", "cross_window_signal_transfer"),
        ("target_tw_manifest", str(Path(args.target_tw_manifest).resolve())),
        ("baseline_feature_names", str(Path(args.baseline_feature_names).resolve())),
        ("source_proxy_features", str(Path(args.source_proxy_features).resolve())),
        ("source_offset_min", args.source_offset_min),
        ("source_offset_max", args.source_offset_max),
        ("source_proxy_tokens", len(source_names)),
        ("source_proxy_features_parsed", len(parsed_source)),
        ("source_proxy_features_in_range", len(source_in_range)),
        (
            "source_proxy_features_filtered_by_offset",
            len(parsed_source) - len(source_in_range),
        ),
        ("source_unique_signals_in_range", source_unique_signals_in_range),
        ("transferred_signals", transferred_signals),
        ("baseline_columns", len(baseline_names)),
        ("added_target_window_columns", len(transfer_names)),
        ("final_columns", len(final_names)),
        ("train_shape", list(train_x.shape)),
        ("test_shape", list(test_x.shape)),
        ("train_buffer", str(train_buffer)),
        ("test_buffer", str(test_buffer)),
        ("feature_names_file", str(feature_names_file)),
        ("source_blocks", blocks),
    ])
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "complete").touch()
    print(
        f"wrote {metadata_path}: baseline={len(baseline_names)}, "
        f"added={len(transfer_names)}, final={len(final_names)}"
    )


if __name__ == "__main__":
    main()
