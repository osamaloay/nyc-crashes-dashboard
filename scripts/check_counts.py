import duckdb, os
print('CWD:', os.getcwd())
for f in ['crash_locations.parquet','crashes_summary.parquet','merged_cleaned_dataset.csv']:
    print(f, 'exists?', os.path.exists(f))
con = duckdb.connect()
try:
    if os.path.exists('crash_locations.parquet'):
        r = con.execute("SELECT COUNT(*) AS cnt FROM parquet_scan('crash_locations.parquet')").fetchone()[0]
        print('locations_rows:', r)
    if os.path.exists('crashes_summary.parquet'):
        r2 = con.execute("SELECT COUNT(*) AS cnt FROM parquet_scan('crashes_summary.parquet')").fetchone()[0]
        print('summary_rows:', r2)
        ssum = con.execute("SELECT SUM(COUNT) FROM parquet_scan('crashes_summary.parquet')").fetchone()[0]
        print('summary_sum_COUNT:', ssum)
    if os.path.exists('merged_cleaned_dataset.csv'):
        csv_cnt = con.execute("SELECT COUNT(*) FROM read_csv_auto('merged_cleaned_dataset.csv')").fetchone()[0]
        print('csv_rows:', csv_cnt)
except Exception as e:
    print('ERROR', e)
finally:
    con.close()
