import pandas as pd
import os
s = os.path.join(os.getcwd(), 'crashes_summary.parquet')
l = os.path.join(os.getcwd(), 'crash_locations.parquet')
print('summary exists?', os.path.exists(s))
print('locations exists?', os.path.exists(l))
if os.path.exists(l):
    loc = pd.read_parquet(l)
    print('locations rows:', len(loc))
if os.path.exists(s):
    sumdf = pd.read_parquet(s)
    print('summary rows:', len(sumdf))
    if 'COUNT' in sumdf.columns:
        print('summary COUNT sum:', int(sumdf['COUNT'].sum()))
    else:
        print('summary has no COUNT')
    # unique VEHICLE_TYPE counts
    if 'VEHICLE_TYPE' in sumdf.columns:
        print('unique VEHICLE_TYPE:', sumdf['VEHICLE_TYPE'].nunique())
