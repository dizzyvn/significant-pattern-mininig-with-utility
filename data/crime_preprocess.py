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
df = pd.read_csv('data/crime/crime.csv', low_memory=False)
columns = ['Crime Name1', 'Place', 'Street Type', 'Start Date/Time']
df = df[columns]
df.columns = ['name1', 'place', 'street', 'start']

# ==================
# Then, we categorize data for each feature
# 1. Crime type
df = df.loc[df['name1'] != 'Not a Crime']  # Remove 'Not a Crime'
name1_dict = {'Crime Against Person': 0,
              'Crime Against Society': 1,
              'Crime Against Property': 2}
df['name1'] = df['name1'].map(name1_dict)
df = df.dropna()
df['name1'] = df['name1'].astype(int)

# ====
# 2. Place
labels = [
    ['Street'],
    ['Residence'],
    ['Parking Lot', 'Parking Garage'],
    ['Retail'],
    ['School/College', 'Library'],
    ['Restaurant'],
    ['Grocery/Supermarket', 'Convenience Store', 'Commercial'],
    ['Gas Station'],
    ['Park'],
    ['Bar/Night Club'],
    ['Bank', 'Bank/S&L/Credit Union', 'Check Cashing Est.'],
    ['Hospital/Emergency Care Center']
]


def get_short_place_name(name):
    """
    Get the short name from long place name
    e.g.,
    'Street - In vehicle' into 'Street'
    'Residence - Single Family' into 'Residence'
    Keyword Arguments:
    name -- original name of the place
    """
    tokens = name.split(' -')
    return tokens[0]


df['place'] = df['place'].apply(get_short_place_name)
label_dict = {}
for i, label in enumerate(labels):
    for c in label:
        label_dict[c] = i

df['place'] = df['place'].map(label_dict)
df = df.dropna()
df['place'] = df['place'].astype(int)

# ====
# 3. Street
labels = [['RD'],
          ['AVE'],
          ['DR'],
          ['ST'],
          ['BLV'],
          ['LA'],
          ['CT'],
          ['WAY'],
          ['PIK'],
          ['TER'],
          ['PL'],
          ['CIR'],
          ['HWY'],
          ['PKW']]

label_dict = {}
for i, label in enumerate(labels):
    for c in label:
        label_dict[c] = i

df['street'] = df['street'].map(label_dict)
df = df.dropna()
df['street'] = df['street'].astype(int)

# ====
# 4. Time
df['start'] = pd.to_datetime(df['start'])
df = df.dropna()
df['start'] = (df['start'].dt.hour.astype(int))
df['start'] = (df['start'] - 12).abs()

# ================
# Create data
columns = ['place', 'street', 'start']
tasks = ['person', 'society', 'property']

for i, task in enumerate(tasks):
    X = df[columns]
    Y = df['name1'] == i

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
            'p': p_calc(n_data_pos, n_data_pat, n_data, n_pos, alternative='two-sided')
        })

    # Dumping the pre-processed data
    with open(f'data/preprocessed/{task}.pkl', 'wb') as f:
        pickle.dump({'N': n_data,
                     'N_pos': n_pos,
                     'datas': data,
                     'family_feats': [0, 1],
                     'utility_feats': [2]},
                    f)
