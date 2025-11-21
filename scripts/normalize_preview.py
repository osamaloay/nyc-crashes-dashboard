import pandas as pd
import re
import os

PAR_SUM = os.path.join(os.getcwd(), 'crashes_summary.parquet')
print('Reading', PAR_SUM)
df = pd.read_parquet(PAR_SUM)

# cleaning functions (copy from app)

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

print('Rows before:', len(df))
if 'VEHICLE_TYPE' in df.columns:
    clean = df['VEHICLE_TYPE'].apply(clean_label).dropna().astype(str)
    print('Unique vehicle types after cleaning:', clean.nunique())
    print('\nTop vehicle types after cleaning:\n')
    print(clean.value_counts().head(100).to_string())
    print('\nVehicle entries containing "van" after cleaning:\n')
    van = clean[clean.str.contains('van', case=False, na=False)]
    print(van.value_counts().to_string())
else:
    print('VEHICLE_TYPE not present')

print('\nSample cleaned rows (first 50):')
if 'VEHICLE_TYPE' in df.columns:
    print(df['VEHICLE_TYPE'].head(50).apply(clean_label).to_string())
