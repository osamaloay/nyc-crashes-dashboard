"""Pre-aggregate person-level summaries from `nyc_crashes.parquet`.

Generates parquet files under `data/person_aggregates/` for the common
person-level queries used by the dashboard (safety vs injury, injuries by hour,
age groups, severity by gender, person-type rates, etc.).

Run:
    python scripts/preaggregate_person.py

This will read `nyc_crashes.parquet` from the repo root. If the file is large
this operation may take time and memory; it's intended for an offline build
step for deployment.
"""
import os
import duckdb
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PQ = os.path.join(ROOT, 'nyc_crashes.parquet')
OUT_DIR = os.path.join(ROOT, 'data', 'person_aggregates')

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(PQ):
    print('nyc_crashes.parquet not found in repo root. Aborting.')
    raise SystemExit(1)

con = duckdb.connect(database=':memory:')
try:
    # Detect actual column names in the parquet (datasets often vary in naming)
    cols = [c for c in con.execute(f"SELECT * FROM parquet_scan('{PQ.replace('\\','/')}') LIMIT 0").df().columns]

    def _choose(candidates):
        """Return the first actual column name that matches any candidate (exact or substring, case-insensitive)."""
        lower_map = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        # substring match
        for cand in candidates:
            for c in cols:
                if cand.lower() in c.lower():
                    return c
        return None

    def _q(colname):
        # Quote identifier safely for DuckDB SQL
        if colname is None:
            return None
        return '"' + colname.replace('"', '""') + '"'

    # Map expected semantic fields to actual column names observed in the parquet
    COL_SAFETY = _choose(['safety_equipment','safety','seat_belt','seatbelt','belt_used'])
    COL_INJURY = _choose(['person_injury','injury','injury_severity','injury_type'])
    COL_CRASH_DT = _choose(['crash_datetime','crash_date_time','crash_date','crash_dt','CRASH_DATETIME'])
    COL_AGE = _choose(['person_age','age','age_of_person'])
    COL_SEX = _choose(['person_sex','sex','gender'])
    COL_PERSON_TYPE = _choose(['person_type','person_type_name','person_role','road_user_type','personclass'])
    COL_EJECTION = _choose(['ejection','ejected'])
    COL_POSITION = _choose(['position_in_vehicle','position','occupant_position','person_position','position_in_vehicle_code','occupant'])
    COL_VEH = _choose(['person_vehicle_type','vehicle_type','vehicle_type_code','vehicle_desc','vehicle'])
    COL_BOROUGH = _choose(['borough'])

    # log mapping for diagnostics
    print('Detected columns mapping:')
    print('  safety:', COL_SAFETY)
    print('  injury:', COL_INJURY)
    print('  crash_datetime:', COL_CRASH_DT)
    print('  age:', COL_AGE)
    print('  sex:', COL_SEX)
    print('  person_type:', COL_PERSON_TYPE)
    print('  ejection:', COL_EJECTION)
    print('  position:', COL_POSITION)
    print('  vehicle:', COL_VEH)
    print('  borough:', COL_BOROUGH)

    # Safety vs Injury
    safety_col = _q(COL_SAFETY) if COL_SAFETY else "''"
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    q = f"""
    SELECT
      lower(trim(replace(replace(COALESCE({safety_col},''),'[',''),']',''))) AS safety,
      coalesce(NULLIF(trim({injury_col}),''),'Unknown') AS injury,
      COUNT(*) AS cnt
    FROM parquet_scan('{PQ.replace('\\','/')}')
    GROUP BY safety, injury
    ORDER BY cnt DESC
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'safety_vs_injury.parquet'), index=False)

    # Injuries by hour
    crash_dt_col = _q(COL_CRASH_DT) if COL_CRASH_DT else None
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    where_dt = f"WHERE {crash_dt_col} IS NOT NULL" if crash_dt_col else ""
    q = f"""
    SELECT EXTRACT(hour FROM TRY_CAST({crash_dt_col} AS TIMESTAMP)) AS hour,
           coalesce(NULLIF(trim({injury_col}),''),'Unknown') AS injury,
           COUNT(*) AS cnt
    FROM parquet_scan('{PQ.replace('\\','/')}')
    {where_dt}
    GROUP BY hour, injury
    ORDER BY hour
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'injuries_by_hour.parquet'), index=False)

    # Age group counts
    age_col = _q(COL_AGE) if COL_AGE else None
    age_expr = f"TRY_CAST({age_col} AS INTEGER)" if age_col else "NULL"
    q = f"""
    SELECT CASE WHEN {age_expr} <= 18 THEN '0-18'
                WHEN {age_expr} <= 30 THEN '19-30'
                WHEN {age_expr} <= 50 THEN '31-50'
                WHEN {age_expr} > 50 THEN '51+'
                ELSE 'Unknown' END AS age_group,
           COUNT(*) AS cnt
    FROM parquet_scan('{PQ.replace('\\','/')}')
    GROUP BY age_group
    ORDER BY cnt DESC
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'age_group_counts.parquet'), index=False)

    # Severity by gender
    sex_col = _q(COL_SEX) if COL_SEX else "''"
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    q = f"""
    SELECT coalesce(NULLIF(trim({sex_col}),'Unknown'),'Unknown') AS sex,
           coalesce(NULLIF(trim({injury_col}),'Unknown'),'Unknown') AS injury,
           COUNT(*) AS cnt
    FROM parquet_scan('{PQ.replace('\\','/')}')
    GROUP BY sex, injury
    ORDER BY cnt DESC
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'severity_by_gender.parquet'), index=False)

    # Person type injury rates
    ptype_col = _q(COL_PERSON_TYPE) if COL_PERSON_TYPE else "''"
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    q = f"""
    SELECT coalesce(NULLIF(trim({ptype_col}),'Unknown'),'Unknown') AS person_type,
           SUM(CASE WHEN {injury_col} IS NOT NULL AND lower(trim({injury_col})) NOT IN ('no injury','none','unknown','') THEN 1 ELSE 0 END) AS injured,
           COUNT(*) AS total
    FROM parquet_scan('{PQ.replace('\\','/')}')
    GROUP BY person_type
    ORDER BY injured DESC
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'person_type_injury_rates.parquet'), index=False)

    # Ejection vs severity
    ejection_col = _q(COL_EJECTION) if COL_EJECTION else "''"
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    q = f"""
    SELECT coalesce(NULLIF(trim({ejection_col}),'Unknown'),'Unknown') AS ejection,
           coalesce(NULLIF(trim({injury_col}),'Unknown'),'Unknown') AS injury,
           COUNT(*) AS cnt
    FROM parquet_scan('{PQ.replace('\\','/')}')
    GROUP BY ejection, injury
    ORDER BY cnt DESC
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'ejection_vs_severity.parquet'), index=False)

    # Position vs severity
    pos_col = _q(COL_POSITION) if COL_POSITION else "''"
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    q = f"""
    SELECT coalesce(NULLIF(trim({pos_col}),'Unknown'),'Unknown') AS position,
           coalesce(NULLIF(trim({injury_col}),'Unknown'),'Unknown') AS injury,
           COUNT(*) AS cnt
    FROM parquet_scan('{PQ.replace('\\','/')}')
    GROUP BY position, injury
    ORDER BY cnt DESC
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'position_vs_severity.parquet'), index=False)

    # Weekday vs weekend injuries
    crash_dt_col = _q(COL_CRASH_DT) if COL_CRASH_DT else None
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    where_dt = f"WHERE {crash_dt_col} IS NOT NULL" if crash_dt_col else ""
    q = f"""
    SELECT CASE WHEN strftime('%w', TRY_CAST({crash_dt_col} AS TIMESTAMP)) IN ('0','6') THEN 'Weekend' ELSE 'Weekday' END AS day_type,
           SUM(CASE WHEN {injury_col} IS NOT NULL AND lower(trim({injury_col})) NOT IN ('no injury','none','unknown','') THEN 1 ELSE 0 END) AS injured
    FROM parquet_scan('{PQ.replace('\\','/')}')
    {where_dt}
    GROUP BY day_type
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'weekday_weekend_injuries.parquet'), index=False)

    # Motorcycle vs car fatalities
    veh_col = _q(COL_VEH) if COL_VEH else None
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    if veh_col:
        q = f"""
        SELECT
            CASE WHEN lower({veh_col}) LIKE '%motor%' OR lower({veh_col}) LIKE '%bike%' THEN 'Motorcycle'
                 WHEN lower({veh_col}) LIKE '%sedan%' OR lower({veh_col}) LIKE '%car%' OR lower({veh_col}) LIKE '%pass%' OR lower({veh_col}) LIKE '%auto%' THEN 'Car'
                 ELSE 'Other' END AS vtype,
            SUM(CASE WHEN lower(COALESCE({injury_col},'')) LIKE '%kill%' OR lower(COALESCE({injury_col},'')) LIKE '%fatal%' THEN 1 ELSE 0 END) AS fatalities,
            COUNT(*) AS total
        FROM parquet_scan('{PQ.replace('\\','/')}')
        GROUP BY vtype
        """
    else:
        # fallback to checking any likely vehicle columns by name
        q = f"""
        SELECT
            'Other' AS vtype,
            SUM(CASE WHEN lower(COALESCE({injury_col},'')) LIKE '%kill%' OR lower(COALESCE({injury_col},'')) LIKE '%fatal%' THEN 1 ELSE 0 END) AS fatalities,
            COUNT(*) AS total
        FROM parquet_scan('{PQ.replace('\\','/')}')
        """
    # Note: many datasets use different vehicle column names; try best-effort fallback
    try:
        df = con.execute(q).df()
    except Exception:
        # fallback: try vehicle_type column name
        q2 = q.replace('person_vehicle_type','vehicle_type')
        df = con.execute(q2).df()
    df.to_parquet(os.path.join(OUT_DIR, 'motorcycle_vs_car_fatalities.parquet'), index=False)

    # Borough vs injury
    borough_col = _q(COL_BOROUGH) if COL_BOROUGH else "''"
    injury_col = _q(COL_INJURY) if COL_INJURY else "''"
    q = f"""
    SELECT coalesce(NULLIF(trim({borough_col}),'Unknown'),'Unknown') AS borough,
           coalesce(NULLIF(trim({injury_col}),'Unknown'),'Unknown') AS injury,
           COUNT(*) AS cnt
    FROM parquet_scan('{PQ.replace('\\','/')}')
    GROUP BY borough, injury
    ORDER BY borough
    """
    df = con.execute(q).df()
    df.to_parquet(os.path.join(OUT_DIR, 'borough_vs_injury.parquet'), index=False)

    print('Wrote person-level aggregates to', OUT_DIR)
finally:
    con.close()
