import os
import time
import traceback

from pathlib import Path

# Try to import plotly and app.figures (load module by path to avoid package import issues)
try:
    import plotly
except Exception as e:
    print('Failed to import plotly:', e)
    raise

import importlib.util
from pathlib import Path as _P
figures_path = _P('app') / 'figures.py'
if not figures_path.exists():
    print('figures module not found at', figures_path)
    raise SystemExit(1)
spec = importlib.util.spec_from_file_location('app_figures', str(figures_path))
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# locate parquet files
par_loc = Path('crash_locations.parquet')
summary_loc = Path('crashes_summary.parquet')
meta_loc = Path('metadata.parquet')

candidates = {
    'crash_locations': par_loc if par_loc.exists() else None,
    'crashes_summary': summary_loc if summary_loc.exists() else None,
    'metadata': meta_loc if meta_loc.exists() else None,
}

for k, v in list(candidates.items()):
    if v is None:
        # attempt to find any match
        matches = list(ROOT.glob(f"**/*{k.replace('_','')}*.parquet"))
        candidates[k] = matches[0] if matches else None

print('Using files:')
for k, v in candidates.items():
    print(f'  {k}:', v)

PAR_LOC = str(candidates['crash_locations']) if candidates['crash_locations'] else None
SUM_LOC = str(candidates['crashes_summary']) if candidates['crashes_summary'] else None
META_LOC = str(candidates['metadata']) if candidates['metadata'] else None

# Define tests: (name, func, kwargs)
tests = [
    ('crashes_by_year', figures.crashes_by_year, {'parquet_path': PAR_LOC, 'boroughs': None, 'year_range': None}),
    ('injuries_by_borough', figures.injuries_by_borough, {'parquet_path': SUM_LOC, 'boroughs': None, 'year_range': None}),
    ('factor_bar', figures.factor_bar, {'parquet_path': 'nyc_crashes.parquet', 'boroughs': None, 'year_range': None, 'top_n': 10}),
    ('map_aggregate_scatter', figures.map_aggregate_scatter, {'parquet_path': PAR_LOC, 'boroughs': None, 'year_range': None, 'sample_limit': 1000, 'cluster_precision': 2}),
    ('temporal_heatmap', figures.temporal_heatmap, {'parquet_path': PAR_LOC, 'boroughs': None, 'year_range': None}),
    ('severity_by_borough', figures.severity_by_borough, {'parquet_path': PAR_LOC, 'boroughs': None, 'year_range': None}),
    ('density_map', figures.density_map, {'parquet_path': PAR_LOC, 'boroughs': None, 'year_range': None}),
    ('gender_pie', figures.gender_pie, {'parquet_path': 'nyc_crashes.parquet', 'boroughs': None, 'year_range': None}),
    ('age_histogram', figures.age_histogram, {'parquet_path': 'nyc_crashes.parquet', 'boroughs': None, 'year_range': None}),
]

results = []

for name, func, kwargs in tests:
    print('\n---')
    print('Running:', name)
    start = time.time()
    try:
        fig = func(**kwargs)
        elapsed = time.time() - start
        ok = fig is not None
        fig_type = type(fig).__name__
        print(f'  OK: returned {fig_type} in {elapsed:.2f}s')
        # quick sanity: ensure it's a Plotly figure
        try:
            import plotly.graph_objs as go
            if not isinstance(fig, go.Figure):
                print('  Warning: returned object is not plotly.graph_objs.Figure')
        except Exception:
            pass
        results.append((name, True, None))
    except Exception as e:
        elapsed = time.time() - start
        print(f'  FAILED in {elapsed:.2f}s')
        traceback.print_exc()
        results.append((name, False, str(e)))

print('\n=== Summary ===')
for name, ok, err in results:
    print(f'{name}:', 'OK' if ok else 'FAILED')

# exit code
failed = [r for r in results if not r[1]]
if failed:
    print('\nSome tests failed.')
    raise SystemExit(2)
else:
    print('\nAll tests passed')
    raise SystemExit(0)
