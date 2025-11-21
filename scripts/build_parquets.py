#!/usr/bin/env python3
"""
Memory-efficient builder for derived parquet files.

This script reads the large source parquet in streaming fashion using
`pyarrow.dataset` and performs incremental aggregation and writing so the
entire dataset never needs to be resident in pandas at once.

Outputs produced:
- crashes_summary.parquet  (exploded by VEHICLE_TYPE and FACTOR with aggregates)
- crash_locations.parquet  (lat/lon subset)
- metadata.parquet         (string-serialized metadata)
- metadata.json            (human readable metadata)

Run:
    python .\scripts\build_parquets.py

Note: requires `pyarrow` and `pandas` installed.
"""
import ast
import os
import json
from collections import defaultdict
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd


SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nyc_crashes.parquet")
OUT_SUM = os.path.join(os.path.dirname(os.path.dirname(__file__)), "crashes_summary.parquet")
OUT_LOC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "crash_locations.parquet")
OUT_META = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata.parquet")
OUT_META_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata.json")


def parse_list_field(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x)
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(t).strip() for t in val if str(t).strip()]
    except Exception:
        # fallback split by comma or semicolon
        parts = []
        for sep in [',', ';', '|']:
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                break
        if not parts:
            parts = [s.strip()]
        return parts


def stream_build(src_path: str):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source parquet not found: {src_path}")

    print("Opening dataset (pyarrow.dataset)...")
    dataset = ds.dataset(src_path, format='parquet')

    # Columns we'll read in streaming fashion to build outputs
    needed_cols = [
        'BOROUGH', 'LATITUDE', 'LONGITUDE', 'FULL ADDRESS', 'CRASH_DATETIME', 'YEAR',
        'SEVERITY_SCORE',
        'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
        'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
        'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
        'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED',
        'ALL_VEHICLE_TYPES', 'ALL_CONTRIBUTING_FACTORS', 'ALL_CONTRIBUTING_FACTORS_STR'
    ]

    # Prepare ParquetWriter for locations (schema chosen from projected fields)
    loc_schema = pa.schema([
        pa.field('LATITUDE', pa.float32()),
        pa.field('LONGITUDE', pa.float32()),
        pa.field('BOROUGH', pa.string()),
        pa.field('YEAR', pa.int32()),
        pa.field('SEVERITY_SCORE', pa.int32()),
        pa.field('FULL ADDRESS', pa.string()),
        pa.field('CRASH_DATETIME', pa.timestamp('ms')),
    ])

    loc_writer = pq.ParquetWriter(OUT_LOC, loc_schema, compression='snappy')

    # Aggregation map: key -> aggregates
    # key: (borough, year, vehicle_type, factor)
    agg = defaultdict(lambda: {'COUNT': 0, 'SUM_INJURED': 0, 'SUM_KILLED': 0, 'SUM_SEVERITY': 0.0})

    meta_boroughs = set()
    meta_years = set()
    meta_vehicle_types = set()
    meta_factors = set()

    # Create a scanner and iterate in batches
    scanner = dataset.scanner(columns=needed_cols, batch_size=64_000)
    print("Streaming batches and aggregating (this may take a while)...")
    for idx, batch in enumerate(scanner.to_batches()):
        df = batch.to_pandas()

        # Ensure YEAR column exists
        if 'YEAR' not in df.columns or df['YEAR'].isna().all():
            if 'CRASH_DATETIME' in df.columns:
                df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
                df['YEAR'] = pd.to_numeric(df['CRASH_DATETIME'].dt.year, errors='coerce').fillna(0).astype('Int32')
            else:
                df['YEAR'] = pd.Series([0] * len(df), dtype='Int32')

        # Compute numeric injury cols safely if present
        inj_cols = [
            'NUMBER OF PERSONS INJURED', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF CYCLIST INJURED', 'NUMBER OF MOTORIST INJURED'
        ]
        kill_cols = [
            'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST KILLED'
        ]
        for c in inj_cols + kill_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
            else:
                df[c] = 0

        # Totals and severity
        df['TOTAL_INJURED'] = df[inj_cols].sum(axis=1)
        df['TOTAL_KILLED'] = df[kill_cols].sum(axis=1)
        df['SEVERITY_SCORE'] = df.get('SEVERITY_SCORE', df['TOTAL_INJURED'] + df['TOTAL_KILLED'] * 5)

        # Prepare and write location rows for those with lat/lon
        loc_df = df.loc[df['LATITUDE'].notnull() & df['LONGITUDE'].notnull(), ['LATITUDE', 'LONGITUDE', 'BOROUGH', 'YEAR', 'SEVERITY_SCORE', 'FULL ADDRESS', 'CRASH_DATETIME']].copy()
        if not loc_df.empty:
            # Downcast types
            loc_df['LATITUDE'] = pd.to_numeric(loc_df['LATITUDE'], errors='coerce').astype('float32')
            loc_df['LONGITUDE'] = pd.to_numeric(loc_df['LONGITUDE'], errors='coerce').astype('float32')
            # CAST YEAR to int32 safely
            try:
                loc_df['YEAR'] = pd.to_numeric(loc_df['YEAR'], errors='coerce').fillna(0).astype('int32')
            except Exception:
                loc_df['YEAR'] = loc_df['YEAR'].astype('Int32').fillna(0).astype('int32')
            # Convert CRASH_DATETIME to pandas datetime then to pyarrow table
            if 'CRASH_DATETIME' in loc_df.columns:
                loc_df['CRASH_DATETIME'] = pd.to_datetime(loc_df['CRASH_DATETIME'], errors='coerce')
            table = pa.Table.from_pandas(loc_df, schema=loc_schema, preserve_index=False)
            loc_writer.write_table(table)

        # For the summary: explode vehicle types and factors and aggregate counts
        # We'll iterate rows in the small batch (fits in memory per batch size)
        for _, row in df.iterrows():
            borough = row.get('BOROUGH') if pd.notna(row.get('BOROUGH')) else 'Unknown'
            year = int(row.get('YEAR') or 0)
            total_inj = int(row.get('TOTAL_INJURED') or 0)
            total_kill = int(row.get('TOTAL_KILLED') or 0)
            severity = float(row.get('SEVERITY_SCORE') or 0.0)

            meta_boroughs.add(borough)
            if year:
                meta_years.add(year)

            veh_cell = row.get('ALL_VEHICLE_TYPES') if 'ALL_VEHICLE_TYPES' in row.index else None
            fac_cell = None
            if 'ALL_CONTRIBUTING_FACTORS' in row.index:
                fac_cell = row.get('ALL_CONTRIBUTING_FACTORS')
            elif 'ALL_CONTRIBUTING_FACTORS_STR' in row.index:
                fac_cell = row.get('ALL_CONTRIBUTING_FACTORS_STR')

            veh_list = parse_list_field(veh_cell)
            fac_list = parse_list_field(fac_cell)

            if not veh_list:
                veh_list = ['Unknown']
            if not fac_list:
                fac_list = ['Unknown']

            for vt in veh_list:
                meta_vehicle_types.add(vt)
                for f in fac_list:
                    meta_factors.add(f)
                    key = (borough, year, vt, f)
                    agg[key]['COUNT'] += 1
                    agg[key]['SUM_INJURED'] += total_inj
                    agg[key]['SUM_KILLED'] += total_kill
                    agg[key]['SUM_SEVERITY'] += severity

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} batches...")

    # Close location writer
    loc_writer.close()

    # Build summary DataFrame from agg dict
    print("Building summary DataFrame from aggregates...")
    rows = []
    for (borough, year, vt, f), v in agg.items():
        cnt = v['COUNT']
        rows.append({
            'BOROUGH': borough,
            'YEAR': int(year) if year else pd.NA,
            'VEHICLE_TYPE': vt,
            'FACTOR': f,
            'COUNT': int(cnt),
            'AVG_SEVERITY': float(v['SUM_SEVERITY'] / cnt) if cnt else 0.0,
            'SUM_INJURED': int(v['SUM_INJURED']),
            'SUM_KILLED': int(v['SUM_KILLED'])
        })

    summary_df = pd.DataFrame(rows)
    # Cast types to smaller ones where possible
    if not summary_df.empty:
        try:
            summary_df['YEAR'] = summary_df['YEAR'].astype('Int32')
            summary_df['COUNT'] = summary_df['COUNT'].astype('int32')
        except Exception:
            pass

    print(f"Writing summary to {OUT_SUM} ({len(summary_df)} rows)...")
    summary_df.to_parquet(OUT_SUM, engine='pyarrow', compression='snappy', index=False)

    # Write metadata.json and metadata.parquet (serialized values)
    meta = {
        'boroughs': sorted(list(meta_boroughs)),
        'years': sorted([int(y) for y in sorted(list(meta_years))]) if meta_years else [],
        'vehicle_types': sorted(list(meta_vehicle_types)),
        'factors': sorted(list(meta_factors)),
        'columns_count': len(dataset.schema)
    }
    print("Writing metadata files...")
    with open(OUT_META_JSON, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    # Serialize meta values as JSON strings for parquet column
    meta_serialized = {k: json.dumps(v, ensure_ascii=False) for k, v in meta.items()}
    meta_df = pd.DataFrame([{'key': k, 'value': v} for k, v in meta_serialized.items()])
    meta_df.to_parquet(OUT_META, engine='pyarrow', compression='snappy', index=False)

    print('Done. Files created:')
    print(' -', OUT_SUM)
    print(' -', OUT_LOC)
    print(' -', OUT_META)
    print(' -', OUT_META_JSON)


if __name__ == '__main__':
    stream_build(SRC)
# scripts/build_parquets.py
import ast
import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

SRC = "nyc_crashes.parquet"
OUT_SUM = "crashes_summary.parquet"
OUT_LOC = "crash_locations.parquet"
OUT_META = "metadata.parquet"

print("Loading source parquet (may use memory) ...")
df = pd.read_parquet(SRC, engine="pyarrow")
print(f"Loaded {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB mem approx")

# --- Normalize key columns & create YEAR ---
df['CRASH_DATETIME'] = pd.to_datetime(df.get('CRASH_DATETIME', pd.NaT), errors='coerce')
df['YEAR'] = df['CRASH_DATETIME'].dt.year.fillna(0).astype('Int32')

# Standardize borough capitalization
if 'BOROUGH' in df.columns:
    df['BOROUGH'] = df['BOROUGH'].astype(str).str.strip().str.upper()
    df['BOROUGH'] = df['BOROUGH'].replace({
        'MANHATTAN': 'Manhattan',
        'BROOKLYN': 'Brooklyn',
        'QUEENS': 'Queens',
        'BRONX': 'Bronx',
        'STATEN ISLAND': 'Staten Island'
    }).astype('category')
else:
    df['BOROUGH'] = pd.Categorical(['Unknown'] * len(df))

# Create TOTAL_INJURED, TOTAL_KILLED (safe)
inj_cols = ['NUMBER OF PERSONS INJURED', 'NUMBER OF PEDESTRIANS INJURED',
            'NUMBER OF CYCLIST INJURED', 'NUMBER OF MOTORIST INJURED']
kill_cols = ['NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS KILLED',
             'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST KILLED']

for c in inj_cols + kill_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int32')
    else:
        df[c] = 0

df['TOTAL_INJURED'] = df[inj_cols].sum(axis=1).astype('int32')
df['TOTAL_KILLED'] = df[kill_cols].sum(axis=1).astype('int32')
df['SEVERITY_SCORE'] = (df['TOTAL_INJURED'] + df['TOTAL_KILLED'] * 5).astype('int32')

# --- Parse ALL_VEHICLE_TYPES into list (safe) ---
def parse_list_field(x):
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(t).strip() for t in val if str(t).strip()]
    except Exception:
        # fallback split by comma
        return [p.strip() for p in s.split(',') if p.strip()]
    return []

df['VEHICLE_TYPES_LIST'] = df.get('ALL_VEHICLE_TYPES', "").apply(parse_list_field)

# --- Parse contributing factors similarly ---
if 'ALL_CONTRIBUTING_FACTORS' in df.columns:
    df['FACTORS_LIST'] = df['ALL_CONTRIBUTING_FACTORS'].apply(parse_list_field)
elif 'ALL_CONTRIBUTING_FACTORS_STR' in df.columns:
    df['FACTORS_LIST'] = df['ALL_CONTRIBUTING_FACTORS_STR'].apply(parse_list_field)
else:
    # fallback to three columns
    parts = []
    for i in [1,2,3]:
        c = f'CONTRIBUTING FACTOR VEHICLE {i}'
        if c in df.columns:
            parts.append(df[c].fillna('').astype(str))
    if parts:
        df['FACTORS_LIST'] = pd.Series([parse_list_field(";".join(x)) for x in zip(*parts)])
    else:
        df['FACTORS_LIST'] = [[] for _ in range(len(df))]

# --- Create crash_locations parquet (keep only needed columns) ---
loc_cols = []
for c in ('LATITUDE','LONGITUDE','BOROUGH','YEAR','SEVERITY_SCORE','FULL ADDRESS','CRASH_DATETIME'):
    if c in df.columns:
        loc_cols.append(c)
locations = df.loc[df['LATITUDE'].notnull() & df['LONGITUDE'].notnull(), loc_cols].copy()
# downcast lat/lon floats
locations['LATITUDE'] = pd.to_numeric(locations['LATITUDE'], errors='coerce').astype('float32')
locations['LONGITUDE'] = pd.to_numeric(locations['LONGITUDE'], errors='coerce').astype('float32')

print("Saving locations parquet ...")
locations.to_parquet(OUT_LOC, engine='pyarrow', compression='snappy', index=False)

# --- Build summary: explode vehicle types and factors so grouping is simple ---
print("Exploding vehicle types and factors to build summary (memory efficient)...")
# explode vehicles
veh_df = df[['BOROUGH','YEAR','VEHICLE_TYPES_LIST','FACTORS_LIST','SEVERITY_SCORE','TOTAL_INJURED','TOTAL_KILLED']].copy()
veh_df = veh_df.explode('VEHICLE_TYPES_LIST')
veh_df['VEHICLE_TYPE'] = veh_df['VEHICLE_TYPES_LIST'].fillna('').astype(str).replace('', 'Unknown').astype('category')
# explode factors as separate rows for counting
veh_df = veh_df.explode('FACTORS_LIST')
veh_df['FACTOR'] = veh_df['FACTORS_LIST'].fillna('').astype(str).replace('', 'Unknown').astype('category')

# group by relevant keys
summary = (veh_df
           .groupby(['BOROUGH','YEAR','VEHICLE_TYPE','FACTOR'])
           .agg(
               COUNT = ('VEHICLE_TYPE','size'),
               AVG_SEVERITY = ('SEVERITY_SCORE','mean'),
               SUM_INJURED = ('TOTAL_INJURED','sum'),
               SUM_KILLED = ('TOTAL_KILLED','sum')
           )
           .reset_index()
)
# cast to smaller types where appropriate
summary['COUNT'] = summary['COUNT'].astype('int32')
summary['YEAR'] = summary['YEAR'].astype('Int32')

print("Saving summary parquet ...")
summary.to_parquet(OUT_SUM, engine='pyarrow', compression='snappy', index=False)

# --- Metadata parquet (lists for dropdowns) ---
print("Saving metadata ...")
meta = {
    'boroughs': sorted(df['BOROUGH'].dropna().unique().tolist()),
    'years': sorted(int(y) for y in sorted(df['YEAR'].dropna().unique().tolist()) if y),
    'vehicle_types': sorted(list({v for sub in df['VEHICLE_TYPES_LIST'] for v in sub})),
    'factors': sorted(list({f for sub in df['FACTORS_LIST'] for f in sub})),
    'columns_count': len(df.columns)
}

# Serialize metadata values to JSON strings so pyarrow doesn't try to mix list/object types in a single column
meta_serialized = {k: json.dumps(v, default=str, ensure_ascii=False) for k, v in meta.items()}
meta_df = pd.DataFrame([
    {'key': k, 'value': v} for k, v in meta_serialized.items()
])
# Save as parquet (strings only) and also write a human-friendly JSON file
meta_df.to_parquet(OUT_META, engine='pyarrow', compression='snappy', index=False)
with open('metadata.json', 'w', encoding='utf-8') as fh:
    json.dump(meta, fh, indent=2, ensure_ascii=False)

print("Done. Files created:")
print(" -", OUT_SUM)
print(" -", OUT_LOC)
print(" -", OUT_META)
