import numpy as np
from scipy import stats


def ttest_p_calc(datas, mu):
    p_values = [stats.ttest_1samp(data, mu)[1] for data in datas]
    return np.array(p_values)
