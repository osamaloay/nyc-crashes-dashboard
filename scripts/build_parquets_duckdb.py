#!/usr/bin/env python3
"""
Build derived parquet files using DuckDB for speed and low memory.

This script reads `nyc_crashes.parquet` via DuckDB and performs SQL
operations to explode vehicle/factor lists and aggregate counts. It
produces:
 - crashes_summary.parquet
 - crash_locations.parquet
 - metadata.json

Run:
  python .\scripts\build_parquets_duckdb.py

Requires: duckdb, pandas
"""
import os
import json
import duckdb
import pandas as pd
import textwrap

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'nyc_crashes.parquet')
OUT_SUM = os.path.join(ROOT, 'crashes_summary.parquet')
OUT_LOC = os.path.join(ROOT, 'crash_locations.parquet')
OUT_META_JSON = os.path.join(ROOT, 'metadata.json')


def run():
    if not os.path.exists(SRC):
        raise FileNotFoundError(f"Source parquet not found: {SRC}")

    con = duckdb.connect(database=':memory:')

    # Write locations parquet (only rows with lat/lon)
    print('Writing locations parquet...')
    loc_select = (
        f"SELECT TRY_CAST(\"LATITUDE\" AS DOUBLE) AS latitude, TRY_CAST(\"LONGITUDE\" AS DOUBLE) AS longitude, "
        f"COALESCE(\"BOROUGH\",'Unknown') AS borough, "
        f"CASE WHEN \"CRASH_DATETIME\" IS NOT NULL THEN EXTRACT(year FROM CAST(\"CRASH_DATETIME\" AS TIMESTAMP)) ELSE NULL END AS year, "
        f"(COALESCE(TRY_CAST(\"NUMBER OF PERSONS INJURED\" AS BIGINT),0)+COALESCE(TRY_CAST(\"NUMBER OF PEDESTRIANS INJURED\" AS BIGINT),0)+COALESCE(TRY_CAST(\"NUMBER OF CYCLIST INJURED\" AS BIGINT),0)+COALESCE(TRY_CAST(\"NUMBER OF MOTORIST INJURED\" AS BIGINT),0) + "
        f"5*(COALESCE(TRY_CAST(\"NUMBER OF PERSONS KILLED\" AS BIGINT),0)+COALESCE(TRY_CAST(\"NUMBER OF PEDESTRIANS KILLED\" AS BIGINT),0)+COALESCE(TRY_CAST(\"NUMBER OF CYCLIST KILLED\" AS BIGINT),0)+COALESCE(TRY_CAST(\"NUMBER OF MOTORIST KILLED\" AS BIGINT),0))) "
        f"AS severity_score, \"FULL ADDRESS\", \"CRASH_DATETIME\" "
        f"FROM parquet_scan('{SRC}') WHERE \"LATITUDE\" IS NOT NULL AND \"LONGITUDE\" IS NOT NULL"
    )
    con.execute(f"COPY ({loc_select}) TO '{OUT_LOC}' (FORMAT PARQUET, COMPRESSION 'snappy')")

    # Use string_split and UNNEST to explode vehicles and factors, then aggregate
    print('Building summary via DuckDB aggregation...')
    summary_sql = textwrap.dedent(f"""
        SELECT borough AS BOROUGH,
            year AS YEAR,
            TRIM(veh) AS VEHICLE_TYPE,
            TRIM(fac) AS FACTOR,
           COUNT(*) AS COUNT,
           AVG(severity_score) AS AVG_SEVERITY,
           SUM(persons_injured + ped_injured + cyc_injured + mot_injured) AS SUM_INJURED,
           SUM(persons_killed + ped_killed + cyc_killed + mot_killed) AS SUM_KILLED
        FROM (
      SELECT
        COALESCE("BOROUGH",'Unknown') AS borough,
        CASE WHEN "CRASH_DATETIME" IS NOT NULL THEN EXTRACT(year FROM CAST("CRASH_DATETIME" AS TIMESTAMP)) ELSE NULL END AS year,
        COALESCE(TRY_CAST("NUMBER OF PERSONS INJURED" AS BIGINT),0) AS persons_injured,
        COALESCE(TRY_CAST("NUMBER OF PEDESTRIANS INJURED" AS BIGINT),0) AS ped_injured,
        COALESCE(TRY_CAST("NUMBER OF CYCLIST INJURED" AS BIGINT),0) AS cyc_injured,
        COALESCE(TRY_CAST("NUMBER OF MOTORIST INJURED" AS BIGINT),0) AS mot_injured,
        COALESCE(TRY_CAST("NUMBER OF PERSONS KILLED" AS BIGINT),0) AS persons_killed,
        COALESCE(TRY_CAST("NUMBER OF PEDESTRIANS KILLED" AS BIGINT),0) AS ped_killed,
        COALESCE(TRY_CAST("NUMBER OF CYCLIST KILLED" AS BIGINT),0) AS cyc_killed,
        COALESCE(TRY_CAST("NUMBER OF MOTORIST KILLED" AS BIGINT),0) AS mot_killed,
        (COALESCE(TRY_CAST("NUMBER OF PERSONS INJURED" AS BIGINT),0)+COALESCE(TRY_CAST("NUMBER OF PEDESTRIANS INJURED" AS BIGINT),0)+COALESCE(TRY_CAST("NUMBER OF CYCLIST INJURED" AS BIGINT),0)+COALESCE(TRY_CAST("NUMBER OF MOTORIST INJURED" AS BIGINT),0) + 5*(COALESCE(TRY_CAST("NUMBER OF PERSONS KILLED" AS BIGINT),0)+COALESCE(TRY_CAST("NUMBER OF PEDESTRIANS KILLED" AS BIGINT),0)+COALESCE(TRY_CAST("NUMBER OF CYCLIST KILLED" AS BIGINT),0)+COALESCE(TRY_CAST("NUMBER OF MOTORIST KILLED" AS BIGINT),0))) AS severity_score,
        COALESCE(CAST("ALL_VEHICLE_TYPES" AS VARCHAR), '') AS all_vehicle_types,
        COALESCE(CAST("ALL_CONTRIBUTING_FACTORS" AS VARCHAR), CAST("ALL_CONTRIBUTING_FACTORS_STR" AS VARCHAR), '') AS all_factors
      FROM parquet_scan('{SRC}')
    ) base,
    UNNEST(string_split(all_vehicle_types, ',')) AS vt(veh),
    UNNEST(string_split(all_factors, ',')) AS ft(fac)
    WHERE TRIM(veh) <> '' AND TRIM(fac) <> ''
    GROUP BY borough, year, TRIM(veh), TRIM(fac)
    """)

    con.execute(f"COPY ({summary_sql}) TO '{OUT_SUM}' (FORMAT PARQUET, COMPRESSION 'snappy')")

    # Extract metadata lists
    print('Extracting metadata...')
    boroughs = con.execute(f"SELECT DISTINCT COALESCE(\"BOROUGH\",'Unknown') FROM parquet_scan('{SRC}') ORDER BY 1").fetchall()
    years = con.execute(f"SELECT DISTINCT EXTRACT(year FROM CAST(\"CRASH_DATETIME\" AS TIMESTAMP)) FROM parquet_scan('{SRC}') WHERE \"CRASH_DATETIME\" IS NOT NULL ORDER BY 1").fetchall()
    vehicle_types = con.execute(
        f"SELECT DISTINCT TRIM(veh) FROM parquet_scan('{SRC}'), UNNEST(string_split(CAST(\"ALL_VEHICLE_TYPES\" AS VARCHAR), ',')) AS vt(veh) WHERE TRIM(veh) <> '' ORDER BY TRIM(veh)"
    ).fetchall()
    factors = con.execute(
        f"SELECT DISTINCT TRIM(fac) FROM parquet_scan('{SRC}'), UNNEST(string_split(CAST(\"ALL_CONTRIBUTING_FACTORS\" AS VARCHAR), ',')) AS ft(fac) WHERE TRIM(fac) <> '' ORDER BY TRIM(fac)"
    ).fetchall()

    def _tuples_to_list(tuples):
        return [t[0] for t in tuples]

    meta = {
        'boroughs': _tuples_to_list(boroughs),
        'years': _tuples_to_list(years),
        'vehicle_types': _tuples_to_list(vehicle_types),
        'factors': _tuples_to_list(factors)
    }

    with open(OUT_META_JSON, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print('Done. Files:')
    print(' -', OUT_SUM)
    print(' -', OUT_LOC)
    print(' -', OUT_META_JSON)

    con.close()


if __name__ == '__main__':
    run()
