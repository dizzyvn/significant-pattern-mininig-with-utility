#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd

from scipy import stats
from tqdm import tqdm


# Helper function that turn numerical values into categorical values
def categorizer(num_data, pivots, labels):
    cat_data = num_data.copy()
    for i in range(len(labels)):
        cat_data[(num_data >= pivots[i]) & (
            num_data < pivots[i+1])] = labels[i]
    return cat_data


# p-value calculator
def p_calc(x_T, n_T, N, N_pos, alternative='greater'):
    cont_table = [[x_T, n_T - x_T],
                  [N_pos - x_T, N - n_T - N_pos + x_T]]
    return stats.fisher_exact(cont_table, alternative=alternative)[1]


# min-p-value calculator
def min_p_calc(n_T, N, N_pos):
    a_s_min = max(0, n_T - N + N_pos)
    a_s_max = min(n_T, N_pos)
    p_s_min = p_calc(a_s_min, n_T, N, N_pos)
    p_s_max = p_calc(a_s_max, n_T, N, N_pos)
    p_s_min = np.inf
    return min(p_s_min, p_s_max)


# We first load the data
df = pd.read_csv('data/crash/crash.csv', low_memory=False)
columns = ['ACRS Report Type',
           'Crash Date/Time',
           'Route Type',
           'Cross-Street Type',
           'Speed Limit',
           'Parked Vehicle',
           'Equipment Problems',
           'Collision Type',
           'Weather',
           'Surface Condition',
           'Light',
           'Traffic Control',
           'Driver At Fault',
           'Driver Distracted By',
           'Injury Severity',
           'Vehicle Year']
df = df[columns]

columns = ['type',
           'time',
           'street-type',
           'cross-type',
           'speed-limit',
           'parked',
           'equip-prob',
           'collision',
           'weather',
           'surface',
           'light',
           'traffic-ctr',
           'driver-fault',
           'driver-distract',
           'injury',
           'vehicle-year']
df.columns = columns


# ==================
# Then, we categorize data for each feature
# 1. Crash type
type_dict = {
    'Property Damage Crash': 0,
    'Injury Crash': 1,
    'Fatal Crash': 1}
df['type'] = df['type'].map(type_dict)
df = df.dropna()
df['type'] = df['type'].astype(int)

# ====
# 2. Street type
labels = [['Maryland (State)'],
          ['County'],
          ['Municipality'],
          ['US (State)'],
          ['Interstate (State)']]
label_dict = {}
for i, label in enumerate(labels):
    for c in label:
        label_dict[c] = i

df['street-type'] = df['street-type'].map(label_dict)
df = df.dropna()
df['street-type'] = df['street-type'].astype(int)

# ====
# 3. Speed limit
labels = [
    [0, 5, 10, 15, 20],
    [25],
    [30],
    [35],
    [40],
    [45],
    [50, 55, 60, 65],
]
label_dict = {}
for i, label in enumerate(labels):
    for c in label:
        label_dict[c] = i
df['speed-limit'] = df['speed-limit'].map(label_dict)
df = df.dropna()
df['speed-limit'] = df['speed-limit'].astype(int)

# ====
# 4. Time
df['time'] = pd.to_datetime(df['time'])
df = df.dropna()
df['time'] = df['time'].dt.hour.astype(int)

# ====
# 5. Traffic control
labels = [
    ['TRAFFIC SIGNAL', 'PERSON', 'FLASHING TRAFFIC SIGNAL',
     'STOP SIGN', 'YIELD SIGN', 'WARNING SIGN',
     'RAILWAY CROSSING DEVICE', 'SCHOOL ZONE SIGN DEVICE'],
    ['NO CONTROLS'],
]
label_dict = {}
for i, label in enumerate(labels):
    for c in label:
        label_dict[c] = i
df['traffic-ctr'] = df['traffic-ctr'].map(label_dict)
df = df.dropna()
df['traffic-ctr'] = df['traffic-ctr'].astype(int)

# ====
# 6. Weather
labels = [
    ['CLEAR'],
    ['CLOUDY'],
    ['RAINING', 'SLEET', 'BLOWING SAND, SOIL, DIRT', 'SEVERE WINDS',
     'SNOW', 'FOGGY', 'WINTRY MIX', 'BLOWING SNOW'],
]
label_dict = {}
for i, label in enumerate(labels):
    for c in label:
        label_dict[c] = i
df['weather'] = df['weather'].map(label_dict)
df = df.dropna()
df['weather'] = df['weather'].astype(int)

# ====
# 7. Surface
labels = [
    ['DRY'],
    ['WET', 'WATER(STANDING/MOVING)', 'SNOW', 'ICE', 'OIL', 
     'MUD, DIRT, GRAVEL', 'SLUSH', 'SAND'],
]
label_dict = {}
for i, label in enumerate(labels):
    for c in label:
        label_dict[c] = i
df['surface'] = df['surface'].map(label_dict)
df = df.dropna()
df['surface'] = df['surface'].astype(int)

# ====
# 8. Light condition
labels = [
    ['DAYLIGHT'],
    ['DARK LIGHTS ON'],
    ['DARK NO LIGHTS']
]
label_dict = {}
for i, label in enumerate(labels):
    for c in label:
        label_dict[c] = i
df['light'] = df['light'].map(label_dict)
df = df.dropna()
df['light'] = df['light'].astype(int)

# ================
# Create data
columns = ['street-type', 'time', 'traffic-ctr',
           'speed-limit', 'weather', 'light']

X = df[columns].copy()
Y = df['type'] == 0

# Create patterns based on the data
# by iteratively appending unique values of each feature
patterns = [([], X, Y)]
for feature in X.columns:
    new_patterns = []
    values = sorted(df[feature].unique())
    for pattern, x, y in patterns:
        for value in values:
            new_y = y[(x[feature] == value)]
            new_x = x[(x[feature] == value)]
            new_pattern = pattern + [value]
            if len(new_x) > 0:
                new_patterns.append((new_pattern, new_x, new_y))
    patterns = new_patterns

# Contigency table
############################
#  n_pos_pat  | ... | n_pos
#  ...        | ... | ...
#  -------------------------
#  n_data_pat | ... | n_data

Y = df['type'] == 1
n_data = len(X)
n_pos = np.sum(Y)

data = []
# We will pre-calculate the min p-value and p-value for each pattern
for pattern, x, y in tqdm(patterns):
    n_data_pat = len(x)
    n_data_pos = np.sum(y)
    data.append({
        'items': pattern,
        'min_p': min_p_calc(n_data_pat, n_data, n_pos),
        'p': p_calc(n_data_pos, n_data_pat, n_data, n_pos)
    })

# Dumping the pre-processed data
with open(f'data/preprocessed/crash.pkl', 'wb') as f:
    pickle.dump({'N': n_data,
                 'N_pos': n_pos,
                 'datas': data,
                 'family_feats': [0, 1],
                 'utility_feats': [2, 3, 4, 5]},
                f)
