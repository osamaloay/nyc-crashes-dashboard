import pandas as pd
import os
p = os.path.join(os.getcwd(), 'crash_locations.parquet')
print('Inspecting', p)
df = pd.read_parquet(p)
print('Rows:', len(df))
print('Columns:', df.columns.tolist())
# try common crash id names
for cid in ['CRASH_ID','CRASH_RECORD_ID','CRASH_ID_','CRASH_KEY','CRASH_RECORD_ID']:
    if cid in df.columns:
        print('Found crash id column:', cid, 'unique count:', df[cid].nunique())
# show sample of ALL_VEHICLE_TYPES and ALL_CONTRIBUTING_FACTORS
for col in ['ALL_VEHICLE_TYPES','ALL_CONTRIBUTING_FACTORS','ALL_VEHICLE_TYPES_clean','ALL_CONTRIBUTING_FACTORS_clean']:
    if col in df.columns:
        print('\nSample values from', col)
        print(df[col].dropna().astype(str).head(20).to_string())
# see any vehicle label variants containing 'Van'
if 'ALL_VEHICLE_TYPES' in df.columns:
    vc = df['ALL_VEHICLE_TYPES'].dropna().astype(str)
    has_van = vc[vc.str.contains('van', case=False, na=False)]
    print('\nNumber of rows with "van" in ALL_VEHICLE_TYPES:', len(has_van))
    print(has_van.head(20).to_string())
else:
    print('ALL_VEHICLE_TYPES not present')
