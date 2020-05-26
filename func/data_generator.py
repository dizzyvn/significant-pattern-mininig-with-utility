import numpy as np
from scipy import stats


def norm_data_generator(
    hypotheses, null_mean=0, alter_mean=0.5, sigma=0.75, n_sample=20
):
    """
    A data generator for t-test
    -- Input:
    hypotheses: Indicator array for true(0)/false(1) hypotheses
    null_mean: mean value for null hypotheses
    alter_mean: mean value for alternative hypotheses
    sigma: standard deviation
    n_sample: number of sample to be generated
    
    -- Output:
    A data array of shape (n_hypotheses, n_sample)
    """
    hypotheses = np.array(hypotheses)
    n = len(hypotheses)

    # parameter assigning
    mu = np.zeros(n)
    mu[np.where(hypotheses == 0)] = null_mean
    mu[np.where(hypotheses == 1)] = alter_mean

    # Data generation
    datas = []
    for i in range(n):
        datas.append(np.random.normal(mu[i], sigma, size=n_sample))

    return np.array(datas)
