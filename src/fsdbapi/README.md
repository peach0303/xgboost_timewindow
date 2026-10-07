# fsdbapi

`fsdbapi` exports bus-level FSDB waveforms directly into the sparse TOG cache
layout used by `code-ai/reduction/dimensionality_reduction_test.py`.

It avoids the FSDB -> CSV -> Python parse path. The C++ exporter uses Verdi
`FsdbReader`, expands bus signals to bit columns, samples levels at fixed FSDB
time tags, converts them to bit-level TOG, and writes SciPy-compatible CSC
NPZ files:

```text
<output-dir>/_work/tog_cache/<dataset>.block00001.tog.cache.csc.npz
<output-dir>/_work/tog_cache_manifest.json
<output-dir>/_work/expanded_layout.jsonl
<output-dir>/original_headers.txt
```

The manifest schema follows the current reduction script's internal
`build_tog_cache_once()` output: `cache_format`, `csv_paths`, `blocks[].names`,
`blocks[].files`, `blocks[].n_rows`, `blocks[].nnz`, and
`blocks[].removed_xz_names`.

## Build

```bash
cd /home/yu.zhang/workspace/fsdbapi
make
```

The default Verdi install is:

```text
/home/yu.zhang/workspace/tools/verdi/verdi/R-2020.12-SP2
```

Override it if needed:

```bash
make VERDI_HOME=/path/to/verdi/R-2020.12-SP2
```

## List Supported Signals

```bash
/home/yu.zhang/workspace/fsdbapi/bin/fsdb_to_tog_cache.py \
  --fsdb /path/to/wave.fsdb \
  --output-dir /tmp/fsdb_cache_probe \
  --start 0 --end 1 --step 1 \
  --list-signals /tmp/fsdb_signals.txt
```

The signal list contains supported 1-byte-per-bit Verilog/VHDL logic-like
signals. Memories, real values, and unsupported value encodings are skipped.

## Export TOG Cache

```bash
/home/yu.zhang/workspace/fsdbapi/bin/fsdb_to_tog_cache.py \
  --fsdb /path/to/wave.fsdb \
  --output-dir /path/to/output/reduction \
  --dataset-name xiangshan_coremark_bit_40000_42000_ns \
  --start 40000 \
  --end 42000 \
  --step 1 \
  --raw-col-block 20000 \
  --signal-list /path/to/signals.txt \
  --tog-mode sample-xor
```

`--signal-list` is optional but recommended for large designs. Each line can be:

- exact bus name, for example `top.core.io_valid[7:0]`
- exact bit name, for example `top.core.io_valid[3]`
- prefix ending in `*`, for example `top.core.backend.*`

`--tog-mode sample-xor` computes `level[row] XOR level[row-1]`, with the first
row set to zero. `--tog-mode any-change` marks a bit as 1 if it changes at least
once within the sample interval.

For datasets whose labels drop the first power sample after even-timestamp
filtering, use the generic sampling controls below. The exporter does not read
power labels; these flags only control the TOG row semantics:

```bash
--end-inclusive --drop-first-tog-row
```

For example, `--start 20000 --end 22000 --step 2 --end-inclusive
--drop-first-tog-row` samples levels at `20000, 20002, ..., 22000`, drops the
first dummy diff row, and writes 1000 TOG rows for transitions ending at
`20002, 20004, ..., 22000`.

FSDB Reader internal `NC`/`NV` values are not treated as x/z. `NC` keeps the
previous value, and `NV`/no-value uses the current/default-zero state to match
`fsdbreport -of b -bit` CSV behavior. Only true `FSDB_BT_VCD_X/Z` can mark a
bit dirty, and only if the sampled level row still sees that X/Z; transient X/Z
events between sample points do not dirty the column.

## Multiple FSDBs For One Global Reduction

For global dimensionality reduction, every FSDB must share one merged cache
manifest. Do not concatenate manifest JSON files by hand. The x/z removal rule
must be applied across all FSDBs jointly, so the merger computes the union of
dirty bit columns and rewrites each per-FSDB CSC NPZ with the same final column
order.

Prepare a file list:

```text
/path/to/a.fsdb
/path/to/b.fsdb
/path/to/c.fsdb
```

Then run:

```bash
python3 /home/yu.zhang/workspace/fsdbapi/bin/fsdb_batch_to_merged_tog_cache.py \
  --fsdb-list /path/to/fsdbs.f \
  --output-dir /path/to/merged/reduction \
  --start 40000 \
  --end 42000 \
  --step 1 \
  --raw-col-block 20000 \
  --signal-list /path/to/signals.txt \
  --tog-mode sample-xor
```

This creates:

```text
/path/to/merged/reduction/_source_exports/<fsdb-stem>/...
/path/to/merged/reduction/_work/tog_cache/*.tog.cache.csc.npz
/path/to/merged/reduction/_work/tog_cache_manifest.json
```

If you already exported each FSDB separately, merge the manifests directly:

```bash
python3 /home/yu.zhang/workspace/fsdbapi/bin/merge_tog_cache_manifests.py \
  --output-dir /path/to/merged/reduction \
  --manifest /path/to/a/reduction/_work/tog_cache_manifest.json \
  --manifest /path/to/b/reduction/_work/tog_cache_manifest.json \
  --manifest /path/to/c/reduction/_work/tog_cache_manifest.json
```

## Output Contract

The generated NPZ files are sparse CSC matrices equivalent to
`scipy.sparse.save_npz(..., compressed=False)` in structure:

- `data.npy`: `uint8`, all ones
- `indices.npy`: `int32`, row indices
- `indptr.npy`: `int64`, CSC column pointer
- `shape.npy`: `int64[2]`
- `format.npy`: `b"csc"`

They are intentionally uncompressed ZIP entries so the C++ tool does not depend
on Python, NumPy, or SciPy.

## Continue Dimensionality Reduction

`code-ai/reduction/dimensionality_reduction_test.py` now accepts an existing
cache manifest:

```bash
python /home/yu.zhang/workspace/code-ai/reduction/dimensionality_reduction_test.py \
  --input-cache-manifest /path/to/output/reduction/_work/tog_cache_manifest.json \
  --output-dir /path/to/output/reduction \
  --output-meta /path/to/output/reduction/meta.json \
  --output-cols /path/to/output/reduction/final.cols.txt \
  --output-original-headers /path/to/output/reduction/original_headers.copy.txt \
  --save-final \
  --keep-cache
```

When `--input-cache-manifest` is used, the script skips CSV reading and starts
from `collect_stats_from_cache()`.
