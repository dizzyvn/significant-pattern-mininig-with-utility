""" Synthesis experiment """
import sys
import collections
import numpy as np
from tqdm import tqdm
from func.data_generator import norm_data_generator
from func.tester import ttest_p_calc
from func.func import open_file, write_list


# ===== THE PROPOSED METHOD ======
def spur(hypotheses, p_values, alpha=0.05):
    """
    A simplized procedure of SPUR when all hypotheses are testable
    -- Input:
    hypotheses: Indicator array for true(0)/false(1) hypotheses
    p_values: p_values
    alpha: The significance level
    """
    n = len(hypotheses)
    p_values = np.copy(p_values)

    # Initialization
    sigma = alpha  # current significant butget
    ignored = np.zeros(p_values.shape)  # indicators ignored hypotheses
    rejected = []  # rejected hypotheses
    last_p = 0  # p_min of the last iteration

    # SPUR
    while 1:
        # step 5: Calculate the sigma
        tau = n - np.sum(ignored)  # See proof of Proposition 5.1
        sigma = sigma / tau + last_p

        # step 6 and 7: Get the index and the p-value of
        # the most significant hypothesis
        min_p_idx = np.argmin(p_values)
        min_p = p_values[min_p_idx]

        # step 8: Reject rule
        if min_p <= sigma:
            # step 9: update sigma
            sigma = sigma - tau * (min_p - last_p) + min_p
            last_p = min_p
            rejected.append(min_p_idx)  # step 11: Reject hypothesis
            # step 12: Masking for ignored hypotheses
            ignored[: min_p_idx + 1] = 1
            p_values[: min_p_idx + 1] = np.infty  # also mask the p-values
        else:
            break

        # If all hypotheses got ignored or rejected
        if np.sum(ignored) == n:
            break

    return np.sort(np.array(rejected))


# ===== BONFERRONI =====
def bonferroni(hypotheses, p_values, alpha=0.05):
    """
    The Bonferroni procedure
    -- Input:
    hypotheses: Indicator array for true(0)/false(1) hypotheses
    p_values: p_values
    alpha: The significance level
    """
    n = len(hypotheses)
    p_values = np.copy(p_values)

    threshold = alpha / n
    rejected = np.where(p_values <= threshold)[0]

    return rejected


# ===== WEIGHTED BONFFERONI =====
def weighted_bonferroni(hypotheses, p_values, weight="linear", alpha=0.05):
    """
    The Bonferroni procedure
    -- Input:
    hypotheses: Indicator array for true(0)/false(1) hypotheses
    p_values: p_values
    alpha: The significance level
    """
    n = len(hypotheses)
    p_values = np.copy(p_values)

    if weight == "linear":
        w = (np.arange(n) + 1).astype(float)

    w /= np.sum(w)
    sigma = alpha * w
    rejected = np.where(p_values <= sigma)[0]

    return rejected


# ===== INVALID SPUR =====
def invalid_spur(hypotheses, p_values, alpha=0.05):
    """
    Invalid version of SPUR where budget are not properly managed   -- Input:
    hypotheses: Indicator array for true(0)/false(1) hypotheses
    p_values: p_values
    alpha: The significance level
    """
    n = len(hypotheses)
    p_values = np.copy(p_values)

    # Initialization
    ignored = np.zeros(p_values.shape)
    rejected = []

    # SPUR
    while 1:
        # Get the "original index" and the p-value of
        # the most significant hypothesis
        min_p_idx = np.argmin(p_values)
        min_p = p_values[min_p_idx]

        # Calculate the sigma
        tau = n - np.sum(ignored)
        sigma = alpha / tau

        # Reject rule
        if min_p <= sigma:
            rejected.append(min_p_idx)
            ignored[: min_p_idx + 1] = 1  # Remove uninteresting hypotheses
            p_values[: min_p_idx + 1] = np.infty  # Also mask the p-values
        else:
            break

        # If all hypotheses got ignored or rejected
        if np.sum(ignored) == n:
            break

    return np.sort(np.array(rejected))


# ===== Experiment setting =====
# exps: List of settings that will be used in the experiment
# where each element contains
# a) the name of the setting (e.g., High usefulness)
# b) the indicators for false null hypotheses
# (e.g., '1' means false null hypotheses)
exps = collections.OrderedDict(
    [
        ("High utility", np.array([0] * 80 + [1] * 20)),
        ("Normal utility", np.array([1, 0, 0, 0, 0] * 20)),
        ("Low utility", np.array([1] * 20 + [0] * 80)),
    ]
)

# methods: List of methods that will be used for each setting
methods = collections.OrderedDict(
    [
        ("SPUR", spur),
        ("Bonferroni", bonferroni),
        ("Weighted-Bonferroni", weighted_bonferroni),
        ("Invalid SPUR", invalid_spur),
    ]
)

# ===== main ======
sub_dir = "exp_1/"
n_rep = 10000
np.random.seed(100)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_rep = int(sys.argv[1])

    for exp_name, exp_hypotheses in exps.items():
        # Create file for output based on name of exp and method
        output_files = {
            method_name: open_file(sub_dir, exp_name, method_name)
            for method_name in methods.keys()
        }

        # Running experiment
        print(f"Running [{exp_name}]")
        for f in output_files.values():
            write_list(f, np.where(exp_hypotheses == 1)[0])

        for i in tqdm(range(n_rep)):
            datas = norm_data_generator(exp_hypotheses)
            p_values = ttest_p_calc(datas, mu=0)

            for method_name, method_func in methods.items():
                rejected = method_func(exp_hypotheses, p_values)
                write_list(output_files[method_name], rejected)

        # Close file
        for f in output_files.values():
            f.close()
