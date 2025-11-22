# app/app_nyc_streamlit.py
import streamlit as st
import pandas as pd
import json
import re
import numpy as np
import plotly.express as px
import duckdb
import os
import sys
from datetime import datetime

# Ensure the repository root is on sys.path so `import app.*` works when
# Streamlit runs this file as a script (Streamlit executes the file directly,
# which can cause the package root to be missing from sys.path).
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from app.figures import (
    injuries_by_borough,
    factor_bar,
    crashes_by_year,
    map_aggregate_scatter,
    gender_pie,
    age_histogram,
    temporal_heatmap,
    severity_by_borough,
    density_map,
    safety_vs_injury,
    injuries_by_hour,
    age_group_counts,
    severity_by_gender,
    person_type_injury_rates,
    ejection_vs_severity,
    position_vs_severity,
    weekday_weekend_injuries,
    motorcycle_vs_car_fatalities,
    borough_vs_injury,
)

st.set_page_config(page_title="NYC Crash Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("NYC Crash Dashboard (fast mode)")

# Dark theme: deep navy background with teal/indigo accents and high-contrast light text
st.markdown(
    """
    <style>
    /* App background and primary containers */
    :root, .stApp { background: #071122; color: #e6eef8; }
    .block-container { background: #0f1b2a; border-radius: 10px; padding: 1.2rem 1.5rem; color: #e6eef8; max-width: 900px; margin: 1.5rem auto; }

    /* ensure there's top padding so the header/title is fully visible */
    header, .stApp, .block-container { padding-top: 1rem; }

    /* prevent the top title from being clipped and give it room */
    h1 { margin-top: 0.25rem !important; margin-bottom: 0.35rem !important; font-size: 36px !important; line-height: 1.12 !important; word-break: break-word; }
    .stPageHeader { padding-top: 0.5rem; }

    /* Headings and primary accents */
    h1, h2, h3 { color: #e6eef8 !important; }
    .stMetricValue, .stMetricLabel { color: #e6eef8 !important; }

    /* Buttons: teal/indigo accents with readable text */
    .stButton>button { background-color: #7c8cff !important; color: #052025 !important; border: 1px solid rgba(255,255,255,0.06) !important; }

    /* Sidebar: slightly darker to separate content and increase width for controls */
    .css-1d391kg, .stSidebar { background-color: #07121a !important; color: #e6eef8 !important; }
    .stSidebar { min-width: 260px !important; }

    /* Links and small accents */
    a { color: #7c8cff !important; }

        /* Improve card contrast and subtle shadows */
        .stCard { box-shadow: 0 6px 18px rgba(2,6,23,0.6); border-radius: 8px; background: #0f2433; }

        /* Tweak metric and header spacing for denser dark layout */
        .stMetric { padding: 8px 10px; }
        .stHeader { color: #e6eef8 !important; }
        .stMetricValue { font-weight: 600; font-size: 20px; }

        /* Make code blocks and tables more readable */
        .stMarkdown code { background: rgba(255,255,255,0.03); color: #dff6fb; }
        .stDataFrame th { color: #e6eef8; }

        /* Responsive: tighten layout on narrow screens */
        @media (max-width: 900px) {
            .block-container { padding-left: 0.75rem; padding-right: 0.75rem; margin: 0.75rem; }
            h1 { font-size: 22px !important; }
            .stSidebar { min-width: 200px !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PAR_SUM = "crashes_summary.parquet"
PAR_LOC = "crash_locations.parquet"
PAR_META_PQ = "metadata.parquet"
PAR_META_JSON = "metadata.json"

@st.cache_data
def load_meta():
    # Prefer JSON metadata produced by the DuckDB pipeline, but fall back to a parquet table if present
    if os.path.exists(PAR_META_JSON):
        try:
            with open(PAR_META_JSON, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            pass
    if os.path.exists(PAR_META_PQ):
        try:
            meta_df = pd.read_parquet(PAR_META_PQ)
            return {r['key']: r['value'] for _, r in meta_df.iterrows()}
        except Exception:
            return {}
    return {}

@st.cache_data
def load_summary():
    df = pd.read_parquet(PAR_SUM)

    # helper to clean single labels
    def clean_label(x):
        if pd.isna(x):
            return x
        s = str(x).strip()
        # remove surrounding brackets/angle-brackets and stray punctuation
        s = re.sub(r'[\[\]\(\)\{\}<>]', '', s)
        # remove stray control punctuation but keep -/& and alphanumerics and spaces and commas
        s = re.sub(r"[^\w\s\-\/,&]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    # helper to clean comma-separated lists
    def clean_list_field(x):
        if pd.isna(x):
            return x
        parts = [clean_label(p) for p in str(x).split(',')]
        parts = [p for p in parts if p]
        return ",".join(parts)

    # Clean VEHICLE_TYPE and FACTOR in the summary if present
    if 'VEHICLE_TYPE' in df.columns:
        df['VEHICLE_TYPE'] = df['VEHICLE_TYPE'].apply(clean_label)
    if 'FACTOR' in df.columns:
        df['FACTOR'] = df['FACTOR'].apply(clean_label)

    # Also clean any aggregated list fields if present
    for col in ['ALL_VEHICLE_TYPES', 'ALL_CONTRIBUTING_FACTORS']:
        if col in df.columns:
            df[col] = df[col].apply(clean_list_field)

    return df

@st.cache_data
def load_locations():
    df = pd.read_parquet(PAR_LOC)
    # Normalize column names to a predictable set the app expects
    cols = {c.lower(): c for c in df.columns}
    # Accept either uppercase LATITUDE/LONGITUDE or lowercase latitude/longitude
    if 'latitude' in cols and 'longitude' in cols:
        df = df.rename(columns={cols['latitude']: 'LATITUDE', cols['longitude']: 'LONGITUDE'})
    elif 'lat' in cols and 'lon' in cols:
        df = df.rename(columns={cols['lat']: 'LATITUDE', cols['lon']: 'LONGITUDE'})
    # Borough/year may be lowercase
    if 'borough' in cols:
        df = df.rename(columns={cols['borough']: 'BOROUGH'})
    if 'year' in cols:
        df = df.rename(columns={cols['year']: 'YEAR'})
    # Clean text fields for vehicle and factor lists so UI choices match
    def clean_label(x):
        if pd.isna(x):
            return x
        s = str(x).strip()
        s = re.sub(r'[\[\]\(\)\{\}<>]', '', s)
        s = re.sub(r"[^\w\s\-\/,&]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def clean_list_field(x):
        if pd.isna(x):
            return x
        parts = [clean_label(p) for p in str(x).split(',')]
        parts = [p for p in parts if p]
        return ",".join(parts)

    for col in ['ALL_VEHICLE_TYPES', 'ALL_CONTRIBUTING_FACTORS']:
        if col in df.columns:
            df[col] = df[col].apply(clean_list_field)
    return df


@st.cache_data
def duckdb_top_vehicles(n=30, boroughs=None, year_range=None):
    # Returns top n vehicle types (cleaned lower-case) using the raw merged dataset if available.
    raw_candidates = ['merged_cleaned_dataset.csv', 'merged_cleaned_dataset.parquet']
    src = None
    for c in raw_candidates:
        p = os.path.join(os.getcwd(), c)
        if os.path.exists(p):
            src = p
            break
    if src is None:
        return pd.DataFrame(columns=['vehicle', 'count'])

    con = duckdb.connect(database=':memory:')
    try:
        if src.lower().endswith('.csv'):
            table_ref = f"read_csv_auto('{src.replace('\\','/')}', header=true)"
        else:
            table_ref = f"parquet_scan('{src.replace('\\','/')}')"

        where_clauses = []
        if boroughs:
            bclean = [b for b in boroughs if b]
            if bclean:
                borough_sql = ','.join(f"'{b.replace("'","''")}'" for b in bclean)
                where_clauses.append(f"BOROUGH IN ({borough_sql})")
        if year_range:
            where_clauses.append(f"EXTRACT(year FROM TRY_CAST(CRASH_DATETIME AS TIMESTAMP)) BETWEEN {int(year_range[0])} AND {int(year_range[1])}")

        where_sql = (' AND '.join(where_clauses)) if where_clauses else '1=1'

        sql = f"""
        SELECT trim(veh) AS vehicle, COUNT(*) AS count
        FROM {table_ref}
        CROSS JOIN UNNEST(string_split(COALESCE(ALL_VEHICLE_TYPES, ''), ',')) AS vt(veh)
        WHERE {where_sql}
        GROUP BY vehicle
        ORDER BY count DESC
        LIMIT {int(n)}
        """
        df = con.execute(sql).df()
    except Exception:
        df = pd.DataFrame(columns=['vehicle', 'count'])
    finally:
        con.close()

    if not df.empty:
        df['vehicle'] = df['vehicle'].astype(str).str.strip()
    return df

try:
    meta = load_meta()
    summary = load_summary()
    locations = load_locations()
except Exception:
    # In hosted environments (Streamlit Cloud) the parquet/csv artifacts
    # may not be present at import time. Fail gracefully to avoid a hard
    # ImportError — initialize empty fallbacks and allow the app to render
    # a helpful message to the user instead.
    meta = {}
    import pandas as _pd
    summary = _pd.DataFrame()
    locations = _pd.DataFrame()

# If metadata JSON was not available, create simple lists from the summary dataframe
if not meta:
    meta = {}
    if not summary.empty:
        meta['boroughs'] = sorted(summary['BOROUGH'].dropna().unique().tolist()) if 'BOROUGH' in summary.columns else []
        meta['years'] = sorted(pd.to_numeric(summary['YEAR'], errors='coerce').dropna().unique().tolist()) if 'YEAR' in summary.columns else []
        meta['vehicle_types'] = sorted(summary['VEHICLE_TYPE'].dropna().unique().tolist()) if 'VEHICLE_TYPE' in summary.columns else []
        meta['factors'] = sorted(summary['FACTOR'].dropna().unique().tolist()) if 'FACTOR' in summary.columns else []

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    boroughs = st.multiselect("Borough", options=meta.get('boroughs', []), default=None)
    years = st.multiselect("Year", options=meta.get('years', []), default=None)
    # Year range slider (falls back to summary/locations if metadata not precise)
    try:
        years_list = sorted([int(y) for y in meta.get('years', [])]) if meta.get('years') else []
        min_year = years_list[0] if years_list else (int(summary['YEAR'].min()) if 'YEAR' in summary.columns else 2010)
        max_year = years_list[-1] if years_list else (int(summary['YEAR'].max()) if 'YEAR' in summary.columns else datetime.now().year)
    except Exception:
        min_year, max_year = 2010, datetime.now().year
    year_range = st.slider('Year range', min_year, max_year, (min_year, max_year))
    vehicles = st.multiselect("Vehicle Type", options=meta.get('vehicle_types', []), default=None)
    factors = st.multiselect("Contributing Factor", options=meta.get('factors', []), default=None)
    injury_slider = st.slider("Minimum total injured", 0, int(summary['SUM_INJURED'].max() if not summary.empty else 0), 0)
    search_text = st.text_input("Search (e.g. 'Brooklyn 2019 bicycle')", "")
    generate = st.button("🔄 Generate Report")

    st.markdown("---")
    st.write("Map options")
    map_points = st.slider("Max map points", 1000, 50000, 20000, step=1000)
    st.write("Display options")
    show_percentages = st.checkbox("Show percentages on pie charts", value=True)
    granularity = st.selectbox('Time granularity', ['Year', 'Month', 'Day'], index=0)
    vehicle_compare = st.multiselect('Compare vehicle types (Year granularity only)', options=meta.get('vehicle_types', []), default=None)
    cluster_map = st.checkbox('Aggregate map points (grid cluster)', value=True)
    cluster_precision = st.slider('Cluster precision (decimal degrees)', 2, 4, 3)
    st.markdown('---')
    st.write('Person-level options')
    person_sample = st.slider('Person-level sample fraction', 0.01, 1.0, 0.05, step=0.01)

# Apply filters on button click or default initial run
def apply_filters(df):
    d = df.copy()
    if boroughs:
        d = d[d['BOROUGH'].isin(boroughs)]
    # apply either explicit years selection or year_range slider
    if years:
        d = d[d['YEAR'].isin(years)]
    else:
        if 'YEAR' in d.columns:
            d = d[(d['YEAR'] >= year_range[0]) & (d['YEAR'] <= year_range[1])]
    if vehicles:
        d = d[d['VEHICLE_TYPE'].isin(vehicles)]
    if factors:
        d = d[d['FACTOR'].isin(factors)]
    if injury_slider:
        d = d[d['SUM_INJURED'] >= injury_slider]
    # basic search: look for words in borough/vehicle/factor fields or year
    if search_text:
        q = search_text.lower()
        d = d[d['BOROUGH'].str.lower().str.contains(q, na=False) |
              d['VEHICLE_TYPE'].str.lower().str.contains(q, na=False) |
              d['FACTOR'].str.lower().str.contains(q, na=False) |
              d['YEAR'].astype(str).str.contains(q, na=False)]
    return d

if generate:
    st.info("Applying filters and generating figures...")
    dfv = apply_filters(summary)

    # Prepare a dataframe of locations for mapping; ensure lat/lon column names
    locs = locations.copy()
    # Accept either LATITUDE/LONGITUDE or lat/lon
    if 'LATITUDE' in locs.columns and 'LONGITUDE' in locs.columns:
        locs['lat'] = locs['LATITUDE']
        locs['lon'] = locs['LONGITUDE']
    else:
        locs['lat'] = locs.get('lat') if 'lat' in locs.columns else locs.get('latitude')
        locs['lon'] = locs.get('lon') if 'lon' in locs.columns else locs.get('longitude')
    if boroughs:
        locs = locs[locs['BOROUGH'].isin(boroughs)] if 'BOROUGH' in locs.columns else locs
    if years:
        locs = locs[locs['YEAR'].isin(years)] if 'YEAR' in locs.columns else locs

    # ensure CRASH_DATETIME is parsed for time granularity calculations
    if 'CRASH_DATETIME' in locs.columns:
        locs['CRASH_DATETIME'] = pd.to_datetime(locs['CRASH_DATETIME'], errors='coerce')
        # populate YEAR from datetime if missing
        if 'YEAR' not in locs.columns or locs['YEAR'].isnull().all():
            locs['YEAR'] = locs['CRASH_DATETIME'].dt.year
        locs['MONTH'] = locs['CRASH_DATETIME'].dt.to_period('M').dt.to_timestamp()
        locs['DAY'] = locs['CRASH_DATETIME'].dt.date

    # helper: on-demand DuckDB vehicle trends (reads original CSV if available)
    def duckdb_vehicle_trend(vehicles, granularity, boroughs, year_range):
        if not vehicles:
            return pd.DataFrame()
        # prefer raw merged CSV if present (one row per crash)
        raw_candidates = ['merged_cleaned_dataset.csv', 'merged_cleaned_dataset.parquet']
        src = None
        for c in raw_candidates:
            if os.path.exists(os.path.join(os.getcwd(), c)):
                src = os.path.join(os.getcwd(), c)
                break
        if src is None:
            # no raw source available; return empty
            return pd.DataFrame()

        unit = {'Year':'year','Month':'month','Day':'day'}.get(granularity, 'month')
        veh_clean = [v.strip() for v in vehicles if v and str(v).strip()]
        if not veh_clean:
            return pd.DataFrame()
        veh_sql = ','.join(f"'{v.replace("'","''").lower()}'" for v in veh_clean)
        borough_sql = None
        if boroughs:
            bclean = [b for b in boroughs if b]
            if bclean:
                borough_sql = ','.join(f"'{b.replace("'","''")}'" for b in bclean)

        con = duckdb.connect(database=':memory:')
        try:
            if src.lower().endswith('.csv'):
                table_ref = f"read_csv_auto('{src.replace('\\','/')}', header=true)"
            else:
                table_ref = f"parquet_scan('{src.replace('\\','/')}')"

            sql = f"""
            SELECT date_trunc('{unit}', TRY_CAST(CRASH_DATETIME AS TIMESTAMP)) AS period,
                   lower(trim(veh)) AS vehicle,
                   COUNT(*) AS count
            FROM {table_ref}
            CROSS JOIN UNNEST(string_split(COALESCE(ALL_VEHICLE_TYPES, ''), ',')) AS vt(veh)
            WHERE lower(trim(veh)) IN ({veh_sql})
            """
            if borough_sql:
                sql += f" AND BOROUGH IN ({borough_sql})\n"
            sql += f" AND EXTRACT(year FROM TRY_CAST(CRASH_DATETIME AS TIMESTAMP)) BETWEEN {int(year_range[0])} AND {int(year_range[1])}\n"
            sql += " GROUP BY period, vehicle ORDER BY period"

            dfq = con.execute(sql).df()
        except Exception as e:
            con.close()
            return pd.DataFrame()
        con.close()
        if not dfq.empty and 'period' in dfq.columns:
            dfq['period'] = pd.to_datetime(dfq['period'])
        return dfq

    # Tabs for different detailed figures
    tab_overview, tab_vehicles, tab_factors, tab_map, tab_time, tab_severity, tab_person = st.tabs(["Overview", "Vehicles", "Factors", "Map", "Time Series", "Severity", "Person-level"])

    with tab_overview:
        st.subheader("Summary numbers")
        # Use the locations table for accurate unique crash counts (avoids double-counting from exploded summary)
        try:
            total_crashes = int(locs.shape[0]) if 'lat' in locs.columns else (int(dfv['COUNT'].sum()) if not dfv.empty else 0)
        except Exception:
            total_crashes = int(dfv['COUNT'].sum()) if not dfv.empty else 0
        total_injured = int(dfv['SUM_INJURED'].sum()) if not dfv.empty else 0
        total_killed = int(dfv['SUM_KILLED'].sum()) if not dfv.empty else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Crashes (approx)", f"{total_crashes:,}")
        col2.metric("Total Injured (agg)", f"{total_injured:,}")
        col3.metric("Total Killed (agg)", f"{total_killed:,}")

        st.markdown("### Crashes by year (filtered)")
        # Use DuckDB-backed figure builder for crashes by year (fast, parquet-backed)
        try:
            fig_year = crashes_by_year(parquet_path=PAR_LOC, boroughs=boroughs or None, year_range=year_range)
            st.plotly_chart(fig_year, width='stretch')
        except Exception:
            # fallback to local pandas-based rendering
            if 'YEAR' in locs.columns and not locs['YEAR'].isnull().all():
                year_df = locs.groupby('YEAR').size().reset_index(name='COUNT')
                fig_year = px.bar(year_df.sort_values('YEAR'), x='YEAR', y='COUNT', labels={'COUNT':'Crashes','YEAR':'Year'}, title='Crashes by Year')
                st.plotly_chart(fig_year, width='stretch')
            else:
                year_df = dfv.groupby('YEAR')['COUNT'].sum().reset_index()
                fig_year = px.bar(year_df.sort_values('YEAR'), x='YEAR', y='COUNT', labels={'COUNT':'Crashes','YEAR':'Year'}, title='Crashes by Year')
                st.plotly_chart(fig_year, width='stretch')

        st.markdown("### Share by borough (filtered)")
        try:
            # injuries are aggregated in the summary parquet (SUM_INJURED/SUM_KILLED)
            fig_b = injuries_by_borough(parquet_path=PAR_SUM, boroughs=boroughs or None, year_range=year_range)
            st.plotly_chart(fig_b, width='stretch')
        except Exception:
            # fallback to pie via pandas
            if 'BOROUGH' in locs.columns:
                borough_df = locs.groupby('BOROUGH').size().reset_index(name='COUNT').sort_values('COUNT', ascending=False)
            else:
                borough_df = dfv.groupby('BOROUGH')['COUNT'].sum().reset_index().sort_values('COUNT', ascending=False)
            if not borough_df.empty:
                fig_b = px.pie(borough_df, values='COUNT', names='BOROUGH', title='Crashes by Borough', hole=0.3)
                if show_percentages:
                    fig_b.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_b, width='stretch')

    with tab_vehicles:
        st.subheader("Top vehicle types")
        # Use DuckDB on-demand aggregation for top vehicles when raw dataset is available; fallback to summary
        top_veh = duckdb_top_vehicles(n=30, boroughs=boroughs, year_range=year_range)
        if top_veh.empty and 'VEHICLE_TYPE' in dfv.columns:
            top_veh = dfv.groupby('VEHICLE_TYPE')['COUNT'].sum().reset_index().sort_values('COUNT', ascending=False).head(30)
            top_veh.columns = ['vehicle','count']

        if not top_veh.empty:
            # ensure column names normalized
            if 'VEHICLE_TYPE' in top_veh.columns and 'COUNT' in top_veh.columns:
                top_veh = top_veh.rename(columns={'VEHICLE_TYPE':'vehicle','COUNT':'count'})
            fig_v = px.bar(top_veh, x='count', y=top_veh.columns[0], orientation='h', title='Top Vehicle Types')
            st.plotly_chart(fig_v, width='stretch')

        st.markdown("### Vehicle trends by year")
        # Vehicle trends: Year granularity uses summary; for Month/Day we can run an on-demand DuckDB query against the raw dataset
        if granularity == 'Year' and 'VEHICLE_TYPE' in dfv.columns and 'YEAR' in dfv.columns:
            veh_pivot = dfv.pivot_table(index='YEAR', columns='VEHICLE_TYPE', values='COUNT', aggfunc='sum', fill_value=0)
            # choose vehicles to display (prefer explicit compare selection)
            if vehicle_compare:
                names = vehicle_compare
            else:
                if 'vehicle' in top_veh.columns:
                    names = top_veh['vehicle'].tolist()[:10]
                elif 'VEHICLE_TYPE' in top_veh.columns:
                    names = top_veh['VEHICLE_TYPE'].tolist()[:10]
                else:
                    names = []
            names = [n for n in names if n in veh_pivot.columns]
            if names:
                veh_trends = veh_pivot[names].reset_index()
                fig_tv = px.line(veh_trends, x='YEAR', y=names, title='Vehicle Type Trends (selected)')
                st.plotly_chart(fig_tv, width='stretch')
        else:
            # Month/Day: run on-demand DuckDB query (reads merged_cleaned_dataset.csv if available)
            vehicles_to_query = vehicle_compare if vehicle_compare else vehicles
            if not vehicles_to_query:
                st.info('Select one or more vehicle types in the sidebar to see Month/Day trends (on-demand query).')
            else:
                with st.spinner('Querying raw dataset for vehicle trends...'):
                    td = duckdb_vehicle_trend(vehicles_to_query, granularity, boroughs or [], year_range)
                if not td.empty:
                    fig_vd = px.line(td, x='period', y='count', color='vehicle', markers=True, title=f'Crashes by {granularity} (selected vehicles)')
                    st.plotly_chart(fig_vd, width='stretch')
                else:
                    st.info('No results from raw dataset for the selected filters/vehicles or raw data file not present.')

    with tab_person:
        st.subheader('Person-level analysis')
        st.markdown('Use sampling to limit scan size when raw person-level tables are large.')

        # Row 1: Safety vs Injury and Injuries by Hour
        r1c1, r1c2 = st.columns([2,3])
        with r1c1:
            st.markdown('**Q1 — Safety equipment vs injury**')
            try:
                fig_safety = safety_vs_injury(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range, sample_frac=person_sample)
                st.plotly_chart(fig_safety, use_container_width=True)
            except Exception as e:
                st.error('Could not build Safety vs Injury chart: ' + str(e))

        with r1c2:
            st.markdown('**Q2 — Injuries by Hour**')
            try:
                fig_hour = injuries_by_hour(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range, sample_frac=person_sample)
                st.plotly_chart(fig_hour, use_container_width=True)
            except Exception as e:
                st.error('Could not build Injuries by Hour chart: ' + str(e))

        # Row 2: Age groups and Gender severity
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown('**Q3 — Age group counts**')
            try:
                fig_ageg = age_group_counts(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_ageg, use_container_width=True)
            except Exception as e:
                st.error('Could not build Age Group chart: ' + str(e))

        with r2c2:
            st.markdown('**Q4 — Severity by Gender**')
            try:
                fig_gender = severity_by_gender(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_gender, use_container_width=True)
            except Exception as e:
                st.error('Could not build Severity by Gender chart: ' + str(e))

        # Row 3: Person type injury rates and Ejection
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            st.markdown('**Q5 — Injury rate by Person Type**')
            try:
                fig_pt = person_type_injury_rates(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_pt, use_container_width=True)
            except Exception as e:
                st.error('Could not build Person Type chart: ' + str(e))

        with r3c2:
            st.markdown('**Q6 — Ejection vs Severity**')
            try:
                fig_ej = ejection_vs_severity(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_ej, use_container_width=True)
            except Exception as e:
                st.error('Could not build Ejection chart: ' + str(e))

        # Row 4: Position vs Severity and Weekday/Weekend
        r4c1, r4c2 = st.columns(2)
        with r4c1:
            st.markdown('**Q7 — Position (Driver vs Passenger)**')
            try:
                fig_pos = position_vs_severity(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_pos, use_container_width=True)
            except Exception as e:
                st.error('Could not build Position chart: ' + str(e))

        with r4c2:
            st.markdown('**Q8 — Weekday vs Weekend Injuries**')
            try:
                fig_wd = weekday_weekend_injuries(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_wd, use_container_width=True)
            except Exception as e:
                st.error('Could not build Weekday/Weekend chart: ' + str(e))

        # Row 5: Motorcycle vs Car fatalities and Borough vs injury
        r5c1, r5c2 = st.columns(2)
        with r5c1:
            st.markdown('**Q9 — Motorcycle vs Car Fatality Rate**')
            try:
                fig_mc = motorcycle_vs_car_fatalities(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_mc, use_container_width=True)
            except Exception as e:
                st.error('Could not build Motorcycle vs Car chart: ' + str(e))

        with r5c2:
            st.markdown('**Q10 — Borough vs Injury Types**')
            try:
                fig_bi = borough_vs_injury(parquet_path='nyc_crashes.parquet', boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_bi, use_container_width=True)
            except Exception as e:
                st.error('Could not build Borough vs Injury chart: ' + str(e))
            vehicles_to_query = vehicle_compare if vehicle_compare else vehicles
            if not vehicles_to_query:
                st.info('Select one or more vehicle types in the sidebar to see Month/Day trends (on-demand query).')
            else:
                with st.spinner('Querying raw dataset for vehicle trends...'):
                    td = duckdb_vehicle_trend(vehicles_to_query, granularity, boroughs or [], year_range)
                if not td.empty:
                    fig_vd = px.line(td, x='period', y='count', color='vehicle', markers=True, title=f'Crashes by {granularity} (selected vehicles)')
                    st.plotly_chart(fig_vd, width='stretch')
                else:
                    st.info('No results from raw dataset for the selected filters/vehicles or raw data file not present.')

    with tab_factors:
        st.subheader("Top contributing factors")
        try:
            # Factors are available in the summary parquet (FACTOR) or in raw person file
            fig_f = factor_bar(parquet_path=PAR_SUM, boroughs=boroughs or None, year_range=year_range, top_n=30)
            st.plotly_chart(fig_f, width='stretch')
        except Exception:
            top_f = dfv.groupby('FACTOR')['COUNT'].sum().reset_index().sort_values('COUNT', ascending=False).head(30)
            fig_f = px.bar(top_f, x='COUNT', y='FACTOR', orientation='h', title='Top Contributing Factors')
            st.plotly_chart(fig_f, width='stretch')

        st.markdown("### Factors by borough (heatmap)")
        if 'FACTOR' in dfv.columns and 'BOROUGH' in dfv.columns:
            fac_pivot = dfv.pivot_table(index='BOROUGH', columns='FACTOR', values='COUNT', aggfunc='sum', fill_value=0)
            # reduce to top factors for readability
            topf = top_f['FACTOR'].tolist()[:12]
            fac_small = fac_pivot[topf]
            fig_h = px.imshow(fac_small, labels=dict(x='Factor', y='Borough', color='Count'), aspect='auto', title='Factors x Borough')
            st.plotly_chart(fig_h, width='stretch')

    with tab_map:
        st.subheader("Map (sampled points)")
        # sample to avoid sending too many points to browser
        locs_sample = locs.dropna(subset=['lat','lon'])
        if len(locs_sample) > map_points:
            locs_sample = locs_sample.sample(map_points, random_state=42)
        if not locs_sample.empty:
            if cluster_map:
                try:
                    fig_map = map_aggregate_scatter(parquet_path=PAR_LOC, boroughs=boroughs or None, year_range=year_range, sample_limit=map_points, cluster_precision=cluster_precision)
                    st.plotly_chart(fig_map, width='stretch')
                except Exception:
                    # fallback to local aggregation
                    prec = cluster_precision
                    locs_agg = locs_sample.copy()
                    locs_agg['lat_r'] = locs_agg['lat'].round(prec)
                    locs_agg['lon_r'] = locs_agg['lon'].round(prec)
                    agg = locs_agg.groupby(['lat_r','lon_r']).agg(count=('lat','size'), severity_mean=('severity_score','mean')).reset_index()
                    agg['lat'] = agg['lat_r']
                    agg['lon'] = agg['lon_r']
                    agg['size'] = (agg['count'] - agg['count'].min() + 1) / (agg['count'].max() - agg['count'].min() + 1) * 30 + 5
                    fig_map = px.scatter_mapbox(agg, lat='lat', lon='lon', size='size', size_max=40, color='severity_mean', hover_data=['count','severity_mean'], zoom=10, height=600)
                    fig_map.update_layout(mapbox_style='open-street-map')
                    st.plotly_chart(fig_map, width='stretch')
            else:
                try:
                    fig_map = px.scatter_mapbox(locs_sample, lat='lat', lon='lon', color='BOROUGH' if 'BOROUGH' in locs_sample.columns else None,
                                                hover_name='BOROUGH' if 'BOROUGH' in locs_sample.columns else None,
                                                hover_data=['severity_score','YEAR'] if 'severity_score' in locs_sample.columns else ['YEAR'],
                                                zoom=10, height=600)
                    fig_map.update_layout(mapbox_style='open-street-map')
                    st.plotly_chart(fig_map, width='stretch')
                except Exception:
                    st.map(locs_sample[['lat','lon']].rename(columns={'lat':'lat','lon':'lon'}))
        else:
            st.info('No location points available for the selected filters.')

    with tab_time:
        st.subheader('Time series (yearly)')
        # Time series according to granularity; for Month/Day we use locations (has CRASH_DATETIME)
        if granularity == 'Year':
            try:
                fig_ts = crashes_by_year(parquet_path=PAR_LOC, boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_ts, width='stretch')
            except Exception:
                if 'YEAR' in dfv.columns:
                    ts = dfv.groupby('YEAR')['COUNT'].sum().reset_index()
                    fig_ts = px.line(ts.sort_values('YEAR'), x='YEAR', y='COUNT', markers=True, title='Crashes per Year')
                    st.plotly_chart(fig_ts, width='stretch')
                else:
                    st.info('Year data not available')
        else:
            try:
                fig_ts = temporal_heatmap(parquet_path=PAR_LOC, boroughs=boroughs or None, year_range=year_range)
                st.plotly_chart(fig_ts, width='stretch')
            except Exception:
                if 'CRASH_DATETIME' in locs.columns:
                    l = locs.dropna(subset=['CRASH_DATETIME'])
                    if boroughs and 'BOROUGH' in l.columns:
                        l = l[l['BOROUGH'].isin(boroughs)]
                    if 'YEAR' in l.columns:
                        l = l[(l['YEAR'] >= year_range[0]) & (l['YEAR'] <= year_range[1])]
                    if granularity == 'Month':
                        ts = l.groupby(pd.Grouper(key='CRASH_DATETIME', freq='M')).size().reset_index(name='COUNT')
                    else:
                        ts = l.groupby(pd.Grouper(key='CRASH_DATETIME', freq='D')).size().reset_index(name='COUNT')
                    if not ts.empty:
                        fig_ts = px.line(ts, x=ts.columns[0], y='COUNT', markers=False, title=f'Crashes per {granularity}')
                        st.plotly_chart(fig_ts, width='stretch')
                    else:
                        st.info('No events for selected filters/timeframe')
                else:
                    st.info('CRASH_DATETIME not present in locations; cannot build Month/Day timeseries')

    with tab_severity:
        st.subheader('Severity analysis')
        try:
            fig_sb = severity_by_borough(parquet_path=PAR_LOC, boroughs=boroughs or None, year_range=year_range)
            st.plotly_chart(fig_sb, width='stretch')
        except Exception:
            if 'AVG_SEVERITY' in dfv.columns:
                sev_b = dfv.groupby('BOROUGH')['AVG_SEVERITY'].mean().reset_index().sort_values('AVG_SEVERITY', ascending=False)
                fig_sb = px.bar(sev_b, x='AVG_SEVERITY', y='BOROUGH', orientation='h', title='Average Severity by Borough')
                st.plotly_chart(fig_sb, width='stretch')

        try:
            fig_density = density_map(parquet_path=PAR_LOC, boroughs=boroughs or None, year_range=year_range)
            st.plotly_chart(fig_density, width='stretch')
        except Exception:
            if 'VEHICLE_TYPE' in dfv.columns:
                sev_v = dfv.groupby('VEHICLE_TYPE')['AVG_SEVERITY'].mean().reset_index().sort_values('AVG_SEVERITY', ascending=False).head(20)
                fig_sv = px.bar(sev_v, x='AVG_SEVERITY', y='VEHICLE_TYPE', orientation='h', title='Avg Severity by Vehicle (top 20)')
                st.plotly_chart(fig_sv, width='stretch')

    # allow download of the filtered summary
    st.markdown('---')
    st.download_button('Download filtered summary (CSV)', dfv.to_csv(index=False).encode('utf-8'), file_name='filtered_summary.csv')

else:
    st.info("Adjust filters and press 'Generate Report' to update charts.")
    st.markdown("**Data snapshot**")
    st.write("Summary rows:", len(summary), "Location rows:", len(locations))
