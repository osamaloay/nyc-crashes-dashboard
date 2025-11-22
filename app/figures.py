"""Reusable Plotly figure builders for NYC crashes dashboard.

These functions prefer to run small DuckDB SQL queries directly against Parquet
artifacts (`crash_locations.parquet`, `crashes_summary.parquet`) so the app can
stay memory-efficient and deployable on Streamlit Cloud.

Each function accepts either a `duckdb.DuckDBPyConnection` via `con` or will
open a temporary in-memory connection. Use the `parquet_path` argument to point
to a Parquet file in the repo root (default names below).

Example usage:
    from app.figures import injuries_by_borough
    fig = injuries_by_borough(parquet_path='crash_locations.parquet')

Notes:
- Functions return Plotly `Figure` objects; the UI layer (Streamlit, Dash)
  should render them with `st.plotly_chart(fig)` or `dcc.Graph(figure=fig)`.
"""

from typing import List, Optional
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Dark Plotly template and borough colors for the app's dark UI
DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#071122",
        plot_bgcolor="#0f1b2a",
        font=dict(color="#e6eef8", family="Inter, Arial, sans-serif"),
        title=dict(font=dict(color="#e6eef8")),
        xaxis=dict(color="#e6eef8", gridcolor="#072433"),
        yaxis=dict(color="#e6eef8", gridcolor="#072433"),
        legend=dict(font=dict(color="#e6eef8")),
        colorway=["#7c8cff", "#0ea5a4", "#06b6d4", "#4ade80", "#60a5fa", "#94d2bd"],
    )
)

# Brighter borough palette for visibility on dark background
BOROUGH_COLORS = {
    'Manhattan': '#7c8cff',
    'Brooklyn': '#06b6d4',
    'Queens': '#4ade80',
    'Bronx': '#60a5fa',
    'Staten Island': '#94d2bd',
    'Unknown': '#9ca3af'
}


def _open_con(con):
    if con is not None:
        return con, False
    return duckdb.connect(database=':memory:'), True


def _detect_col_fn(parquet_path: str, con: duckdb.DuckDBPyConnection):
    """Return a function col(name) -> actual column name present in the parquet.

    It queries zero rows to discover available columns, and matches case-insensitively.
    """
    try:
        cols = [c for c in con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 0").df().columns]
    except Exception:
        # fallback: return identity lower-casing
        return lambda n: n.lower()
    cols_lu = {c.lower(): c for c in cols}

    # common synonyms mapping for columns that may differ between summary/location files
    synonyms = {
        'total_injured': ['sum_injured', 'total_injured', 'injured_total', 'sum_injured'],
        'all_contributing_factors': ['all_contributing_factors', 'contributing_factors', 'factors', 'factor'],
        'person_sex': ['person_sex', 'sex', 'gender'],
        'person_age': ['person_age', 'age'],
        'day_of_week': ['day_of_week', 'dayofweek', 'dow'],
        'hour': ['hour', 'crash_hour', 'h'],
        'latitude': ['latitude', 'lat'],
        'longitude': ['longitude', 'lon', 'lng'],
        'severity_score': ['severity_score', 'severity'],
        'borough': ['borough'],
        'crash_datetime': ['crash_datetime', 'CRASH_DATETIME']
    }

    def col(name: str) -> str:
        # try exact match first
        key = name.lower()
        if key in cols_lu:
            return cols_lu[key]
        # try synonyms
        if key in synonyms:
            for cand in synonyms[key]:
                if cand.lower() in cols_lu:
                    return cols_lu[cand.lower()]
        # fallback: return lowercase name (may error later)
        return name.lower()

    return col


def _build_where(boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None, col_fn=None) -> str:
    clauses = []
    if col_fn is None:
        # identity
        col_fn = lambda n: n
    if boroughs:
        safe = ",".join(f"'{b.replace("'","''")}'" for b in boroughs)
        clauses.append(f"{col_fn('BOROUGH')} IN ({safe})")
    if year_range and len(year_range) == 2:
        dt_col = col_fn('CRASH_DATETIME')
        clauses.append(f"EXTRACT(year FROM TRY_CAST({dt_col} AS TIMESTAMP)) BETWEEN {int(year_range[0])} AND {int(year_range[1])}")
    return " AND ".join(clauses) if clauses else '1=1'


def injuries_by_borough(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
                        boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None):
    """Return a bar chart of total injuries by borough (Query runs in DuckDB)."""
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        where = _build_where(boroughs, year_range, col_fn=col)
        borough_col = col('BOROUGH')
        inj_col = col('TOTAL_INJURED')
        sql = f"""
        SELECT {borough_col} AS BOROUGH,
               SUM(COALESCE({inj_col}, 0)) AS TOTAL_INJURED
        FROM parquet_scan('{parquet_path}')
        WHERE {where}
        GROUP BY {borough_col}
        ORDER BY TOTAL_INJURED DESC
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.bar(df, x='BOROUGH', y='TOTAL_INJURED', labels={'TOTAL_INJURED': 'Total Injured', 'BOROUGH': 'Borough'},
                 color='BOROUGH', color_discrete_map=BOROUGH_COLORS)
    fig.update_layout(template=DARK_TEMPLATE, margin=dict(t=30, b=20), showlegend=False)
    return fig


def factor_bar(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
               boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None, top_n: int = 15):
    """Bar chart of top contributing factors (explodes list field with DuckDB)."""
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        where = _build_where(boroughs, year_range, col_fn=col)
        factors_col = col('ALL_CONTRIBUTING_FACTORS')
        # if the parquet already contains a 'factor' column (summary), use it
        cols = [c.lower() for c in con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 0").df().columns]
        if 'factor' in cols:
            sql = f"""
            SELECT factor, COUNT(*) AS cnt
            FROM parquet_scan('{parquet_path}')
            WHERE {where} AND factor IS NOT NULL
            GROUP BY factor
            ORDER BY cnt DESC
            LIMIT {int(top_n)}
            """
        else:
            sql = f"""
            SELECT factor, COUNT(*) AS cnt
            FROM (
                SELECT trim(f) AS factor
                FROM parquet_scan('{parquet_path}'),
                     UNNEST(string_split(COALESCE({factors_col}, ''), ',')) AS t(f)
                WHERE {where}
            )
            GROUP BY factor
            ORDER BY cnt DESC
            LIMIT {int(top_n)}
            """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    df = df.rename(columns={'cnt': 'Count', 'factor': 'Factor'})
    fig = px.bar(df, x='Count', y='Factor', orientation='h', color='Count', color_continuous_scale='purples')
    fig.update_layout(template=DARK_TEMPLATE, margin=dict(t=20, b=20))
    return fig


def crashes_by_year(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
                    boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None):
    """Line chart of crashes per year (grouped by borough when present)."""
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        where = _build_where(boroughs, None, col_fn=col)
        dt_col = col('CRASH_DATETIME')
        borough_col = col('BOROUGH')
        year_expr = f"EXTRACT(year FROM TRY_CAST({dt_col} AS TIMESTAMP))::INT"
        sql = f"""
        SELECT {year_expr} AS YEAR,
               {borough_col} AS BOROUGH,
               COUNT(*) AS crashes
        FROM parquet_scan('{parquet_path}')
        WHERE {where}
        GROUP BY {year_expr}, {borough_col}
        ORDER BY {year_expr}
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.line(df, x='YEAR', y='crashes', color='BOROUGH', markers=True, color_discrete_map=BOROUGH_COLORS)
    fig.update_layout(template=DARK_TEMPLATE, margin=dict(t=20, b=20))
    return fig


def map_aggregate_scatter(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
                          boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                          sample_limit: int = 20000, cluster_precision: int = 3):
    """Aggregate location points into a grid (rounding) and return a scatter map figure.

    This avoids sending millions of points to the browser while still showing hotspots.
    """
    con, close = _open_con(con)
    try:
         col = _detect_col_fn(parquet_path, con)
         where = _build_where(boroughs, year_range, col_fn=col)
         lat_col = col('LATITUDE')
         lon_col = col('LONGITUDE')
         sev_col = col('SEVERITY_SCORE')
         sql = f"""
         SELECT round({lat_col}, {cluster_precision}) AS lat_r,
             round({lon_col}, {cluster_precision}) AS lon_r,
             COUNT(*) AS cnt,
             AVG(COALESCE({sev_col},0)) AS severity_mean
         FROM parquet_scan('{parquet_path}')
         WHERE {where} AND {lat_col} IS NOT NULL AND {lon_col} IS NOT NULL
         GROUP BY lat_r, lon_r
         ORDER BY cnt DESC
         LIMIT {int(sample_limit)}
         """
         df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    df = df.rename(columns={'lat_r': 'lat', 'lon_r': 'lon', 'cnt': 'count', 'severity_mean': 'severity_mean'})
    df['size'] = (df['count'] - df['count'].min() + 1) / (df['count'].max() - df['count'].min() + 1) * 30 + 5
    fig = px.scatter_mapbox(df, lat='lat', lon='lon', size='size', color='severity_mean', hover_data=['count','severity_mean'], zoom=10, height=600)
    fig.update_layout(mapbox_style='open-street-map', template=DARK_TEMPLATE, margin=dict(t=0,b=0))
    return fig


def gender_pie(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
               boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None):
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        where = _build_where(boroughs, year_range, col_fn=col)
        sex_col = col('PERSON_SEX')
        sql = f"""
        SELECT {sex_col} AS sex, COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {where}
        GROUP BY {sex_col}
        ORDER BY cnt DESC
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.pie(df, names='sex', values='cnt', color_discrete_sequence=['#4A90E2', '#FF8DA1', '#95A5A6'])
    fig.update_layout(template=DARK_TEMPLATE, margin=dict(t=20,b=20))
    return fig


def age_histogram(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
                  boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None):
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        where = _build_where(boroughs, year_range, col_fn=col)
        age_col = col('PERSON_AGE')
        # Try to read PERSON_AGE values (may be missing); cast to double
        sql = f"""
        SELECT CAST({age_col} AS DOUBLE) AS age
        FROM parquet_scan('{parquet_path}')
        WHERE {where} AND {age_col} IS NOT NULL
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty or df['age'].dropna().empty:
        return go.Figure()
    fig = px.histogram(df, x='age', nbins=30, marginal='box', color_discrete_sequence=['#FF6B6B'])
    fig.update_layout(template=DARK_TEMPLATE, margin=dict(t=20,b=20), xaxis_title='Age')
    return fig


def temporal_heatmap(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
                     boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None):
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        base_where = _build_where(boroughs, year_range, col_fn=col)
        dt_col = col('CRASH_DATETIME')
        day_expr = f"strftime('%A', TRY_CAST({dt_col} AS TIMESTAMP))"
        hour_expr = f"EXTRACT(hour FROM TRY_CAST({dt_col} AS TIMESTAMP))"
        sql = f"""
        SELECT {day_expr} AS DAY_OF_WEEK, {hour_expr} AS HOUR, COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {base_where} AND {dt_col} IS NOT NULL
        GROUP BY {day_expr}, {hour_expr}
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    # Ensure day ordering
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['DAY_OF_WEEK'] = pd.Categorical(df['DAY_OF_WEEK'], categories=day_order, ordered=True)
    df = df.sort_values(['DAY_OF_WEEK','HOUR'])
    fig = px.density_heatmap(df, x='HOUR', y='DAY_OF_WEEK', z='cnt', color_continuous_scale='viridis')
    fig.update_layout(template=DARK_TEMPLATE, margin=dict(t=20,b=20))
    return fig


def severity_by_borough(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
                        boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None):
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        where = _build_where(boroughs, year_range, col_fn=col)
        borough_col = col('BOROUGH')
        sev_col = col('SEVERITY_SCORE')
        sql = f"""
        SELECT {borough_col} AS BOROUGH, AVG(COALESCE({sev_col},0)) AS avg_sev
        FROM parquet_scan('{parquet_path}')
        WHERE {where}
        GROUP BY {borough_col}
        ORDER BY avg_sev DESC
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.bar(df, x='avg_sev', y='BOROUGH', orientation='h', color='BOROUGH', color_discrete_map=BOROUGH_COLORS)
    fig.update_layout(template=DARK_TEMPLATE, margin=dict(t=20,b=20), showlegend=False)
    return fig


def density_map(parquet_path: str = 'crash_locations.parquet', con: duckdb.DuckDBPyConnection = None,
                boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None):
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        where = _build_where(boroughs, year_range, col_fn=col)
        lat_col = col('LATITUDE')
        lon_col = col('LONGITUDE')
        sev_col = col('SEVERITY_SCORE')
        sql = f"""
        SELECT {lat_col} AS LATITUDE, {lon_col} AS LONGITUDE, COALESCE({sev_col},0) AS severity
        FROM parquet_scan('{parquet_path}')
        WHERE {where} AND {lat_col} IS NOT NULL AND {lon_col} IS NOT NULL
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.density_mapbox(df, lat='LATITUDE', lon='LONGITUDE', z='severity', radius=20, zoom=9, height=500, mapbox_style='open-street-map', color_continuous_scale='viridis')
    fig.update_layout(template=DARK_TEMPLATE, margin=dict(t=10,b=10))
    return fig
