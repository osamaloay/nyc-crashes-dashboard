Place your data files in the project root (same directory as `requirements.txt`):

- `crash_locations.parquet`  — canonical one-row-per-crash dataset used by most figures
- `crashes_summary.parquet`  — pre-aggregated exploded summary (optional)
- `merged_cleaned_dataset.csv` — optional raw CSV used for on-demand DuckDB queries

Usage (from Streamlit or Dash app):

- Import the figure builders:

```python
from app.figures import injuries_by_borough, map_aggregate_scatter, crashes_by_year

fig = injuries_by_borough(parquet_path='crash_locations.parquet')
st.plotly_chart(fig, use_container_width=True)
```

- The figure functions prefer to run DuckDB queries directly against Parquet files (no full-file pandas load).
- If you need clustering via scikit-learn you can enable it, but to keep the app lightweight for Streamlit Cloud we used a simple grid-aggregation fallback by default.

Deployment notes:

- Keep `duckdb` in `requirements.txt` (already present). Avoid heavy compiled packages like `pyarrow` or `fastparquet` in the Cloud build.
- Ensure Parquet artifacts are in the repo or accessible by the deployed app; large parquet files may be hosted externally (S3) and referenced by path/URL.
