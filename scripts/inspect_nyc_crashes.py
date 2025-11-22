import duckdb
from pathlib import Path
p = Path('nyc_crashes.parquet')
if not p.exists():
    print('nyc_crashes.parquet not found at', p.resolve())
    raise SystemExit(1)
con = duckdb.connect()
df = con.execute(f"SELECT * FROM parquet_scan('{p.as_posix()}') LIMIT 0").df()
print('columns:', df.columns.tolist())
