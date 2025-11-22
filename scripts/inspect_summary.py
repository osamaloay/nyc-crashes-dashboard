import duckdb
from pathlib import Path
p = Path('crashes_summary.parquet')
if not p.exists():
    print('crashes_summary.parquet not found at', p.resolve())
    raise SystemExit(1)
con = duckdb.connect()
df = con.execute(f"SELECT * FROM parquet_scan('{p.as_posix()}') LIMIT 0").df()
print('columns:', df.columns.tolist())
