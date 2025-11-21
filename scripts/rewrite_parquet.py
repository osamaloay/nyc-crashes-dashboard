#!/usr/bin/env python3
"""
rewrite_parquet.py

Rewrite an existing Parquet file into a ZSTD-compressed, dictionary-encoded
Parquet dataset with optional partitioning and tuned row-group size.

This script reads the input Parquet by row-group (pyarrow) to avoid loading
the entire file into memory, converts selected columns to categorical (Pandas)
so the output can use dictionary encoding, and writes chunks to a partitioned
dataset using `pyarrow.parquet.write_to_dataset`.

Usage (run locally):
  python scripts/rewrite_parquet.py \
       --input nyc_crashes.parquet \
       --out-dir parquet_zstd_by_year \
       --partition-col YEAR \
       --categorical-cols BOROUGH "VEHICLE TYPE CODE 1" \
       --row-group-rows 200000

Notes:
- Requires `pyarrow` installed (and ideally built with ZSTD support).
- The script writes many files under `out-dir` (one per partition + fragment).
- Test on a small sample first.
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def sizeof(path: Path) -> int:
    """Return total size in bytes for a file or directory."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except Exception:
                pass
    return total


def rewrite_parquet(
    input_path: str,
    out_dir: str,
    partition_cols: list,
    categorical_cols: list,
    chunk_row_target: int = 200_000,
    compression: str = "ZSTD",
    use_dictionary: bool = True,
):
    input_path = Path(input_path)
    out_dir = Path(out_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if out_dir.exists():
        raise FileExistsError(f"Output directory {out_dir} already exists. Remove it or specify a different path.")

    out_dir.mkdir(parents=True, exist_ok=False)

    pqfile = pq.ParquetFile(str(input_path))
    num_row_groups = pqfile.num_row_groups
    print(f"Input: {input_path} | row-groups: {num_row_groups}")

    total_written = 0
    rg_idx = 0
    start_time = time.time()

    while rg_idx < num_row_groups:
        acc = []
        got = 0
        while rg_idx < num_row_groups and got < chunk_row_target:
            try:
                tbl = pqfile.read_row_group(rg_idx)
                df = tbl.to_pandas()
            except Exception as e:
                print(f"Warning: skipping row-group {rg_idx} due to read error: {e}")
                rg_idx += 1
                continue

            acc.append(df)
            got += len(df)
            rg_idx += 1

        if not acc:
            break

        chunk_df = pd.concat(acc, ignore_index=True)

        # Ensure YEAR exists (derive from CRASH_DATETIME) so partitioning works
        if "YEAR" not in chunk_df.columns:
            try:
                chunk_df["CRASH_DATETIME"] = pd.to_datetime(chunk_df.get("CRASH_DATETIME", pd.NaT), errors="coerce")
                chunk_df["YEAR"] = chunk_df["CRASH_DATETIME"].dt.year
            except Exception:
                chunk_df["YEAR"] = pd.NA

        # Convert provided columns to categorical to encourage dictionary encoding
        for c in (categorical_cols or []):
            if c in chunk_df.columns:
                try:
                    chunk_df[c] = chunk_df[c].astype("category")
                except Exception:
                    # fallback: leave as-is
                    pass

        # Write chunk to partitioned dataset
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(out_dir),
            partition_cols=partition_cols or None,
            use_dictionary=use_dictionary,
            compression=compression,
        )

        total_written += len(chunk_df)
        elapsed = time.time() - start_time
        print(f"Wrote chunk: {len(chunk_df):,} rows (total {total_written:,}) — rg_idx {rg_idx}/{num_row_groups} — {elapsed:.1f}s")

    total_time = time.time() - start_time
    in_size = sizeof(input_path)
    out_size = sizeof(out_dir)
    print("\nDone.")
    print(f"Total rows written (approx): {total_written:,}")
    print(f"Input size: {in_size / 1024**2:.2f} MB")
    print(f"Output dataset size: {out_size / 1024**2:.2f} MB")
    print(f"Elapsed: {total_time:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input Parquet file")
    p.add_argument("--out-dir", required=True, help="Directory to write rewritten Parquet dataset")
    p.add_argument("--partition-col", action="append", default=[], help="Column(s) to partition by (repeatable)")
    p.add_argument("--categorical-cols", action="append", default=[], help="Columns to convert to categorical to encourage dictionary encoding (repeatable)")
    p.add_argument("--chunk-rows", type=int, default=200000, help="Approx rows per chunk (controls how many row-groups are aggregated per write)")
    p.add_argument("--compression", default="ZSTD", help="Compression codec to use (ZSTD, BROTLI, SNAPPY)")
    args = p.parse_args()

    rewrite_parquet(
        input_path=args.input,
        out_dir=args.out_dir,
        partition_cols=args.partition_col,
        categorical_cols=args.categorical_cols,
        chunk_row_target=args.chunk_rows,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
