import pandas as pd

PAR_SUM = '../crashes_summary.parquet'
# If running from repo root, adjust path
import os
p = os.path.join(os.getcwd(), 'crashes_summary.parquet')
if os.path.exists(p):
    PAR_SUM = p

print('Reading', PAR_SUM)
df = pd.read_parquet(PAR_SUM)
print('Rows:', len(df))
if 'VEHICLE_TYPE' in df.columns:
    vc = df['VEHICLE_TYPE'].dropna().astype(str)
    print('Unique vehicle types:', vc.nunique())
    print('\nTop vehicle types:\n')
    print(vc.value_counts().head(100).to_string())
    print('\nVehicle entries containing "van":\n')
    van = vc[vc.str.contains('van', case=False, na=False)]
    print(van.value_counts().to_string())
else:
    print('VEHICLE_TYPE column not present in summary')

if 'ALL_VEHICLE_TYPES' in df.columns:
    av = df['ALL_VEHICLE_TYPES'].dropna().astype(str)
    print('\nSample ALL_VEHICLE_TYPES rows (first 20):')
    print(av.head(20).to_string())
else:
    print('ALL_VEHICLE_TYPES not in summary')
