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
import os

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


def _quote_ident(name: str) -> str:
    """Return a safely quoted identifier for SQL (double-quote, escape internal quotes)."""
    if not isinstance(name, str):
        name = str(name)
    return '"' + name.replace('"', '""') + '"'


def _detect_col_fn(parquet_path: str, con: duckdb.DuckDBPyConnection):
    """Return a function col(name) -> actual column name present in the parquet.

    It queries zero rows to discover available columns, and matches case-insensitively.
    """
    try:
        cols = [c for c in con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 0").df().columns]
    except Exception:
        # fallback: return identity lower-casing
        return lambda n: n.lower()
    # Build a normalization map that removes non-alphanumeric chars so
    # columns like 'POSITION IN VEHICLE' or 'Position-In-Vehicle' match
    import re
    def _norm(s: str) -> str:
        return re.sub(r'[^0-9a-z]', '', s.lower())

    cols_lu = {c.lower(): c for c in cols}
    cols_norm = {_norm(c): c for c in cols}

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
            'crash_datetime': ['crash_datetime', 'CRASH_DATETIME'],
            'safety_equipment': ['safety_equipment', 'safety', 'safety_device'],
            'person_injury': ['person_injury', 'injury', 'injury_severity', 'injury_level'],
            'person_type': ['person_type', 'personrole', 'type'],
            'crash_time': ['crash_time', 'time'],
            'crash_date': ['crash_date', 'date'],
            'person_age': ['person_age', 'age'],
            'person_sex': ['person_sex', 'sex', 'gender'],
            'ejection': ['ejection', 'ejected'],
            'position_in_vehicle': ['position_in_vehicle', 'position'],
            'number_of_persons_killed': ['number_of_persons_killed', 'num_killed', 'persons_killed'],
            'vehicle_type': ['vehicle_type', 'veh_type', 'VEHICLE_TYPE']
    }

    def col(name: str) -> str:
        # try exact match first
        key = name.lower()
        if key in cols_lu:
            return cols_lu[key]
        # try normalized exact match (strip spaces/punctuation)
        nk = _norm(name)
        if nk in cols_norm:
            return cols_norm[nk]
        # try synonyms
        if key in synonyms:
            for cand in synonyms[key]:
                ck = cand.lower()
                if ck in cols_lu:
                    return cols_lu[ck]
                if _norm(cand) in cols_norm:
                    return cols_norm[_norm(cand)]
        # last ditch: try to match any column whose normalized form contains the normalized key
        for ncol, orig in cols_norm.items():
            if nk and nk in ncol:
                return orig
        # fallback: return the original name (may error later)
        return name

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
        # honor the year_range filter when provided
        where = _build_where(boroughs, year_range, col_fn=col)
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


def safety_vs_injury(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                     boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                     sample_frac: Optional[float] = None, extra_filter_sql: str = ''):
    """Compare safety equipment usage vs injury categories (stacked bars).

    Returns a stacked bar chart of counts of `person_injury` by `safety_equipment`.
    """
    # Prefer preaggregated parquet if available and no extra SQL filters are requested
    agg_path = os.path.join('data', 'person_aggregates', 'safety_vs_injury.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            df['safety'] = df['safety'].astype(str)
            fig = px.bar(df, x='safety', y='cnt', color='injury', title='Safety Equipment vs Injury (counts)')
            fig.update_layout(template=DARK_TEMPLATE, barmode='stack', xaxis_title='Safety equipment', yaxis_title='Count')
            return fig
        except Exception:
            # fall back to live query
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        safety = col('safety_equipment')
        injury = col('person_injury')
        safety_col = _quote_ident(safety)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        sample_clause = ''
        if sample_frac and 0 < sample_frac < 1:
            # DuckDB: use random() for sampling
            sample_clause = f" AND random() < {float(sample_frac)}"

        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT
            coalesce(NULLIF(lower(trim(replace(replace({safety_col}, '[', ''), ']', ''))),''),'Unknown') AS safety,
            coalesce(NULLIF(trim({injury_col}),'') ,'Unknown') AS injury,
            COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {sample_clause} {extra_clause}
        GROUP BY safety, injury
        ORDER BY cnt DESC
        LIMIT 1000
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    df['safety'] = df['safety'].astype(str)
    fig = px.bar(df, x='safety', y='cnt', color='injury', title='Safety Equipment vs Injury (counts)',
                 color_discrete_sequence=None)
    fig.update_layout(template=DARK_TEMPLATE, barmode='stack', xaxis_title='Safety equipment', yaxis_title='Count')
    return fig


def injuries_by_hour(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                     boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                     sample_frac: Optional[float] = None, extra_filter_sql: str = ''):
    """Counts of injuries by hour of day (0-23)."""
    agg_path = os.path.join('data', 'person_aggregates', 'injuries_by_hour.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            fig = px.bar(df, x='hour', y='cnt', color='injury', title='Injuries by Hour of Day')
            fig.update_layout(template=DARK_TEMPLATE, barmode='stack', xaxis=dict(dtick=1))
            return fig
        except Exception:
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        dt = col('crash_datetime')
        injury = col('person_injury')
        dt_col = _quote_ident(dt)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        sample_clause = ''
        if sample_frac and 0 < sample_frac < 1:
            sample_clause = f" AND random() < {float(sample_frac)}"

        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT EXTRACT(hour FROM TRY_CAST({dt_col} AS TIMESTAMP)) AS hour,
               coalesce(NULLIF(trim({injury_col}),''),'Unknown') AS injury,
               COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {sample_clause} {extra_clause} AND {dt_col} IS NOT NULL
        GROUP BY hour, injury
        ORDER BY hour
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    # show stacked counts per hour
    fig = px.bar(df, x='hour', y='cnt', color='injury', title='Injuries by Hour of Day')
    fig.update_layout(template=DARK_TEMPLATE, barmode='stack', xaxis=dict(dtick=1))
    return fig


def age_group_counts(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                     boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                     extra_filter_sql: str = ''):
    """Counts by age groups (0-18, 19-30, 31-50, 51+)."""
    agg_path = os.path.join('data', 'person_aggregates', 'age_group_counts.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            fig = px.bar(df.sort_values('age_group'), x='age_group', y='cnt', title='Crashes by Age Group')
            fig.update_layout(template=DARK_TEMPLATE)
            return fig
        except Exception:
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        age = col('person_age')
        age_col = _quote_ident(age)
        where = _build_where(boroughs, year_range, col_fn=col)
        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT CASE WHEN TRY_CAST({age_col} AS INTEGER) <= 18 THEN '0-18'
                WHEN TRY_CAST({age_col} AS INTEGER) <= 30 THEN '19-30'
                WHEN TRY_CAST({age_col} AS INTEGER) <= 50 THEN '31-50'
                WHEN TRY_CAST({age_col} AS INTEGER) > 50 THEN '51+'
                ELSE 'Unknown' END AS age_group,
               COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {extra_clause}
        GROUP BY age_group
        ORDER BY cnt DESC
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.bar(df.sort_values('age_group'), x='age_group', y='cnt', title='Crashes by Age Group')
    fig.update_layout(template=DARK_TEMPLATE)
    return fig


def severity_by_gender(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                       boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                       extra_filter_sql: str = ''):
    """Compare injury categories by sex/gender."""
    agg_path = os.path.join('data', 'person_aggregates', 'severity_by_gender.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            fig = px.bar(df, x='sex', y='cnt', color='injury', title='Injury Severity by Gender')
            fig.update_layout(template=DARK_TEMPLATE, barmode='stack')
            return fig
        except Exception:
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        sex = col('person_sex')
        injury = col('person_injury')
        sex_col = _quote_ident(sex)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT coalesce(NULLIF(trim({sex_col}),'Unknown'),'Unknown') AS sex,
               coalesce(NULLIF(trim({injury_col}),'Unknown'),'Unknown') AS injury,
               COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {extra_clause}
        GROUP BY sex, injury
        ORDER BY cnt DESC
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.bar(df, x='sex', y='cnt', color='injury', title='Injury Severity by Gender')
    fig.update_layout(template=DARK_TEMPLATE, barmode='stack')
    return fig


def person_type_injury_rates(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                              boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                              extra_filter_sql: str = ''):
    """Compare injury rates (percent injured) across person types (Pedestrian, Bicyclist, Occupant, etc.)."""
    agg_path = os.path.join('data', 'person_aggregates', 'person_type_injury_rates.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            df['rate'] = df['injured'] / df['total']
            fig = px.bar(df.sort_values('rate', ascending=False), x='person_type', y='rate', title='Injury Rate by Person Type', labels={'rate':'Injury rate'})
            fig.update_layout(template=DARK_TEMPLATE)
            return fig
        except Exception:
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        ptype = col('person_type')
        injury = col('person_injury')
        ptype_col = _quote_ident(ptype)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT coalesce(NULLIF(trim({ptype_col}),'Unknown'),'Unknown') AS person_type,
               SUM(CASE WHEN {injury_col} IS NOT NULL AND lower(trim({injury_col})) NOT IN ('no injury','none','unknown','') THEN 1 ELSE 0 END) AS injured,
               COUNT(*) AS total
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {extra_clause}
        GROUP BY person_type
        ORDER BY injured DESC
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    df['rate'] = df['injured'] / df['total']
    fig = px.bar(df.sort_values('rate', ascending=False), x='person_type', y='rate', title='Injury Rate by Person Type', labels={'rate':'Injury rate'})
    fig.update_layout(template=DARK_TEMPLATE)
    return fig


def ejection_vs_severity(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                         boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                         extra_filter_sql: str = ''):
    """Show distribution of injury categories by ejection status."""
    agg_path = os.path.join('data', 'person_aggregates', 'ejection_vs_severity.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            fig = px.bar(df, x='ejection', y='cnt', color='injury', title='Ejection vs Injury Severity')
            fig.update_layout(template=DARK_TEMPLATE, barmode='stack')
            return fig
        except Exception:
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        eject = col('ejection')
        injury = col('person_injury')
        eject_col = _quote_ident(eject)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT coalesce(NULLIF(trim({eject_col}),'Unknown'),'Unknown') AS ejection,
               coalesce(NULLIF(trim({injury_col}),'Unknown'),'Unknown') AS injury,
               COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {extra_clause}
        GROUP BY ejection, injury
        ORDER BY cnt DESC
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.bar(df, x='ejection', y='cnt', color='injury', title='Ejection vs Injury Severity')
    fig.update_layout(template=DARK_TEMPLATE, barmode='stack')
    return fig


def position_vs_severity(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                         boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                         extra_filter_sql: str = ''):
    """Compare injury categories for Driver vs Passenger positions."""
    agg_path = os.path.join('data', 'person_aggregates', 'position_vs_severity.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            fig = px.bar(df, x='position', y='cnt', color='injury', title='Position in Vehicle vs Injury Severity')
            fig.update_layout(template=DARK_TEMPLATE, barmode='stack')
            return fig
        except Exception:
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        pos = col('position_in_vehicle')
        injury = col('person_injury')
        pos_col = _quote_ident(pos)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT coalesce(NULLIF(trim({pos_col}),'Unknown'),'Unknown') AS position,
               coalesce(NULLIF(trim({injury_col}),'Unknown'),'Unknown') AS injury,
               COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {extra_clause}
        GROUP BY position, injury
        ORDER BY cnt DESC
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.bar(df, x='position', y='cnt', color='injury', title='Position in Vehicle vs Injury Severity')
    fig.update_layout(template=DARK_TEMPLATE, barmode='stack')
    return fig


def weekday_weekend_injuries(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                             boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                             extra_filter_sql: str = ''):
    """Compare injured crash counts on Weekday vs Weekend."""
    agg_path = os.path.join('data', 'person_aggregates', 'weekday_weekend_injuries.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            fig = px.bar(df, x='day_type', y='injured', title='Weekday vs Weekend Injuries')
            fig.update_layout(template=DARK_TEMPLATE)
            return fig
        except Exception:
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        dt = col('crash_datetime')
        injury = col('person_injury')
        dt_col = _quote_ident(dt)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT CASE WHEN strftime('%w', TRY_CAST({dt_col} AS TIMESTAMP)) IN ('0','6') THEN 'Weekend' ELSE 'Weekday' END AS day_type,
               SUM(CASE WHEN {injury_col} IS NOT NULL AND lower(trim({injury_col})) NOT IN ('no injury','none','unknown','') THEN 1 ELSE 0 END) AS injured
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {extra_clause} AND {dt_col} IS NOT NULL
        GROUP BY day_type
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.bar(df, x='day_type', y='injured', title='Weekday vs Weekend Injuries')
    fig.update_layout(template=DARK_TEMPLATE)
    return fig


def motorcycle_vs_car_fatalities(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                                 boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                                 extra_filter_sql: str = ''):
    """Compare fatality counts and rates for motorcycles vs cars.

    This uses simple keyword matching on `vehicle_type` to detect motorcycles.
    """
    agg_path = os.path.join('data', 'person_aggregates', 'motorcycle_vs_car_fatalities.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            df['fatality_rate'] = df['fatalities'] / df['total']
            fig = px.bar(df, x='vtype', y='fatality_rate', title='Fatality Rate: Motorcycle vs Car', labels={'fatality_rate':'Fatalities per person'})
            fig.update_layout(template=DARK_TEMPLATE)
            return fig
        except Exception:
            pass
    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        vcol = col('vehicle_type')
        injury = col('person_injury')
        vcol_col = _quote_ident(vcol)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT
            CASE WHEN lower({vcol_col}) LIKE '%motor%' OR lower({vcol_col}) LIKE '%bike%' THEN 'Motorcycle'
                 WHEN lower({vcol_col}) LIKE '%sedan%' OR lower({vcol_col}) LIKE '%car%' OR lower({vcol_col}) LIKE '%pass%' OR lower({vcol_col}) LIKE '%auto%' THEN 'Car'
                 ELSE 'Other' END AS vtype,
            SUM(CASE WHEN lower(COALESCE({injury_col},'')) LIKE '%kill%' OR lower(COALESCE({injury_col},'')) LIKE '%fatal%' THEN 1 ELSE 0 END) AS fatalities,
            COUNT(*) AS total
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {extra_clause}
        GROUP BY vtype
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    df['fatality_rate'] = df['fatalities'] / df['total']
    fig = px.bar(df, x='vtype', y='fatality_rate', title='Fatality Rate: Motorcycle vs Car', labels={'fatality_rate':'Fatalities per person'})
    fig.update_layout(template=DARK_TEMPLATE)
    return fig


def borough_vs_injury(parquet_path: str = 'nyc_crashes.parquet', con: duckdb.DuckDBPyConnection = None,
                      boroughs: Optional[List[str]] = None, year_range: Optional[List[int]] = None,
                      extra_filter_sql: str = ''):
    """Show distribution of injury types by borough (stacked bars).

    Accepts `extra_filter_sql` so person-level filters can be applied when called
    from the dashboard. When `extra_filter_sql` is empty and a preaggregated
    parquet exists, the function will read the preaggregate for speed.
    """
    # Prefer preaggregate when available and no extra SQL filtering requested
    agg_path = os.path.join('data', 'person_aggregates', 'borough_vs_injury.parquet')
    if not extra_filter_sql and os.path.exists(agg_path):
        try:
            df = pd.read_parquet(agg_path)
            if df.empty:
                return go.Figure()
            fig = px.bar(df, x='borough', y='cnt', color='injury', title='Injury Types by Borough')
            fig.update_layout(template=DARK_TEMPLATE, barmode='stack')
            return fig
        except Exception:
            pass

    con, close = _open_con(con)
    try:
        col = _detect_col_fn(parquet_path, con)
        bcol = col('borough')
        injury = col('person_injury')
        bcol_col = _quote_ident(bcol)
        injury_col = _quote_ident(injury)
        where = _build_where(boroughs, year_range, col_fn=col)
        extra_clause = f" AND ({extra_filter_sql})" if extra_filter_sql else ''
        sql = f"""
        SELECT coalesce(NULLIF(trim({bcol_col}),'Unknown'),'Unknown') AS borough,
               coalesce(NULLIF(trim({injury_col}),'Unknown'),'Unknown') AS injury,
               COUNT(*) AS cnt
        FROM parquet_scan('{parquet_path}')
        WHERE {where} {extra_clause}
        GROUP BY borough, injury
        ORDER BY borough
        """
        df = con.execute(sql).df()
    finally:
        if close:
            con.close()

    if df.empty:
        return go.Figure()
    fig = px.bar(df, x='borough', y='cnt', color='injury', title='Injury Types by Borough')
    fig.update_layout(template=DARK_TEMPLATE, barmode='stack')
    return fig
