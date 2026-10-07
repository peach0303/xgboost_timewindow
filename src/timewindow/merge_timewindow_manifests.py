#!/usr/bin/env python3
"""Merge time-window manifests produced by split dsub jobs."""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict


def parse_csv(text: str):
    return [x.strip() for x in text.split(',') if x.strip()]


def read_filelist(path: str) -> list[str]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                out.append(s)
    return out


def write_filelist(path: str, items: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(item + '\n')


def unique_extend(dst: list[str], src: list[str]) -> None:
    seen = set(dst)
    for item in src:
        if item not in seen:
            dst.append(item)
            seen.add(item)


def merge_unique_combos(loaded, args) -> OrderedDict:
    out = OrderedDict((k, v) for k, v in loaded[0][1].items() if k != 'combinations')
    if args.offsets and 'offsets' in out:
        out['offsets'] = [int(x) for x in parse_csv(args.offsets)]
    if args.data_types and 'data_types' in out:
        out['data_types'] = parse_csv(args.data_types)
    out['jobs'] = len(loaded)

    merged = OrderedDict()
    for path, manifest in loaded:
        combos = manifest.get('combinations', {})
        for combo_key, combo_info in combos.items():
            if combo_key in merged:
                raise SystemExit(f'ERROR: duplicate combo {combo_key} from {path}; use --merge-step1-filelists for Step1 case parts')
            merged[combo_key] = combo_info
    return finalize_order(out, merged, args.combo_order)


def merge_step1_filelists(loaded, args) -> OrderedDict:
    output_root = os.path.dirname(os.path.abspath(args.output))
    out = OrderedDict((k, v) for k, v in loaded[0][1].items() if k not in ('combinations', 'train_stems', 'test_stems'))
    if args.offsets and 'offsets' in out:
        out['offsets'] = [int(x) for x in parse_csv(args.offsets)]
    if args.data_types and 'data_types' in out:
        out['data_types'] = parse_csv(args.data_types)
    out['jobs'] = len(loaded)

    train_stems: list[str] = []
    test_stems: list[str] = []
    grouped: dict[str, dict[str, list[str] | list[int] | dict]] = OrderedDict()

    for _path, manifest in loaded:
        unique_extend(train_stems, manifest.get('train_stems', []))
        unique_extend(test_stems, manifest.get('test_stems', []))
        for combo_key, combo_info in manifest.get('combinations', {}).items():
            fd = combo_info['filelist_dir']
            if combo_key not in grouped:
                grouped[combo_key] = {
                    'meta': combo_info,
                    'train_feature': [],
                    'train_label': [],
                    'train_weight': [],
                    'test_feature': [],
                    'test_label': [],
                    'test_weight': [],
                    'rows': [],
                }
            g = grouped[combo_key]
            unique_extend(g['train_feature'], read_filelist(os.path.join(fd, 'train_feature.f')))
            unique_extend(g['train_label'], read_filelist(os.path.join(fd, 'train_label.f')))
            unique_extend(g['train_weight'], read_filelist(os.path.join(fd, 'train_weight.f')))
            unique_extend(g['test_feature'], read_filelist(os.path.join(fd, 'test_feature.f')))
            unique_extend(g['test_label'], read_filelist(os.path.join(fd, 'test_label.f')))
            unique_extend(g['test_weight'], read_filelist(os.path.join(fd, 'test_weight.f')))
            if 'valid_rows_per_file' in combo_info:
                g['rows'].append(int(combo_info['valid_rows_per_file']))

    merged = OrderedDict()
    for combo_key, g in grouped.items():
        combo_dir = os.path.join(output_root, combo_key)
        os.makedirs(combo_dir, exist_ok=True)
        write_filelist(os.path.join(combo_dir, 'train_feature.f'), g['train_feature'])
        write_filelist(os.path.join(combo_dir, 'train_label.f'), g['train_label'])
        write_filelist(os.path.join(combo_dir, 'test_feature.f'), g['test_feature'])
        write_filelist(os.path.join(combo_dir, 'test_label.f'), g['test_label'])
        write_filelist(os.path.join(combo_dir, 'train_weight.f'), g['train_weight'])
        write_filelist(os.path.join(combo_dir, 'test_weight.f'), g['test_weight'])
        rows = g['rows']
        meta = OrderedDict(g['meta'])
        meta['train_files'] = len(g['train_feature'])
        meta['test_files'] = len(g['test_feature'])
        meta['valid_rows_per_file'] = int(rows[0]) if rows else 0
        meta['valid_rows_min'] = int(min(rows)) if rows else 0
        meta['valid_rows_max'] = int(max(rows)) if rows else 0
        meta['filelist_dir'] = os.path.abspath(combo_dir)
        merged[combo_key] = meta

    out['train_stems'] = train_stems
    out['test_stems'] = test_stems
    return finalize_order(out, merged, args.combo_order)


def finalize_order(out: OrderedDict, merged: OrderedDict, combo_order: str) -> OrderedDict:
    if combo_order:
        ordered = OrderedDict()
        for combo_key in parse_csv(combo_order):
            if combo_key not in merged:
                raise SystemExit(f'ERROR: missing combo {combo_key} in merged manifests')
            ordered[combo_key] = merged[combo_key]
        for combo_key in merged:
            if combo_key not in ordered:
                ordered[combo_key] = merged[combo_key]
        merged = ordered
    out['combinations'] = merged
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Merge time-window Step1/Step2 manifests')
    ap.add_argument('--output', required=True)
    ap.add_argument('--offsets', default='')
    ap.add_argument('--data-types', default='')
    ap.add_argument('--combo-order', default='')
    ap.add_argument('--merge-step1-filelists', action='store_true',
                    help='Allow duplicate combo keys and merge Step1 train/test filelists')
    ap.add_argument('manifests', nargs='+')
    args = ap.parse_args()

    loaded = []
    for path in args.manifests:
        with open(path, 'r', encoding='utf-8') as f:
            loaded.append((path, json.load(f, object_pairs_hook=OrderedDict)))
    if not loaded:
        raise SystemExit('ERROR: no manifests to merge')

    if args.merge_step1_filelists:
        out = merge_step1_filelists(loaded, args)
    else:
        out = merge_unique_combos(loaded, args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"merged {len(out['combinations'])} combinations -> {args.output}")


if __name__ == '__main__':
    main()
