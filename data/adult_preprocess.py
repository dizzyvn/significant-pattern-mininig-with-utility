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
# and use both train and test data
adult_train = pd.read_csv('data/adult/adult.data.csv', header=None)
adult_test = pd.read_csv('data/adult/adult.test.csv', skiprows=1, header=None)
df = pd.concat([adult_train, adult_test])
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income']
df.columns = columns

# ==================
# Then, we categorize data for each feature
# 1. Occupation
occupation_dict = {}
for i, label in enumerate(df['occupation'].unique()):
    occupation_dict[label] = i
df['occupation'] = df['occupation'].map(occupation_dict)
df = df.dropna()
df['occupation'] = df['occupation'].astype(int)

# ====
# 2. Workclass
wk_class = [[' Private'], [' Self-emp-not-inc'], [' Self-emp-inc'],
            [' Local-gov'], [' State-gov'], [' Federal-gov']]
wk_class = {i: c for i, c in enumerate(wk_class)}
wk_dict = {}
for c, labels in wk_class.items():
    for label in labels:
        wk_dict[label] = c
df['workclass'] = df['workclass'].map(wk_dict)
df = df.dropna()
df['workclass'] = df['workclass'].astype(int)

# ====
# 3. Education
edu_class = [
    ['Preschool', '1st-4th'],
    ['5th-6th', '7th-8th', '9th'],
    ['10th', '11th', '12th'],
    ['HS-grad'],
    ['Prof-school'],
    ['Assoc-acdm', 'Assoc-voc'],
    ['Some-college'],
    ['Bachelors'],
    ['Masters'],
    ['Doctorate']
]
edu_class = {i: c for i, c in enumerate(edu_class)}
edu_dict = {}
for c, labels in edu_class.items():
    for label in labels:
        edu_dict[f' {label}'] = c
df['education'] = df['education'].map(edu_dict)

# ====
# 4. Sex
df['sex'] = df['sex'].map({' Male': 0, ' Female': 1})

# ====
# 5. Hour per week
thresholds = [0, 20, 30, 40, 50, 60, 168]
labels = range(len(thresholds) - 1)
df['hours-per-week'] = categorizer(df['hours-per-week'], thresholds, labels)

# ====
# 6. Income
df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})

# ================
# Create data
columns = ['sex', 'workclass', 'occupation', 'hours-per-week', 'education']
for col in columns:
    df[col] = df[col].astype(int)

X = df[columns].copy()
Y = df['income'].copy()

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
        'p': p_calc(n_data_pos, n_data_pat, n_data, n_pos)
    })

# Dumping the pre-processed data
with open(f'data/preprocessed/adult.pkl', 'wb') as f:
    pickle.dump({'N': n_data,
                 'N_pos': n_pos,
                 'datas': data,
                 'family_feats': [0, 1, 2],
                 'utility_feats': [3, 4]},
                f)
