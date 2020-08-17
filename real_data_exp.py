import sys
import os
import pickle
import numpy as np

from collections import OrderedDict
from matplotlib import pyplot as plt

global utility_feats


def get_files(dataset):
    data_dir = f"./data/{dataset}/"
    data_files = []
    for _, _, files in os.walk(data_dir):
        data_files += [(f.split(".")[0], f"{data_dir}{f}")
                       for f in files if "pkl" in f]
    return data_files


def plot_threshold(task, trace):
    """
    Keyword Arguments:
    trace -- trace data of SPUR
    """
    steps, min_ps, thresholds, n_testables = zip(*trace)

    # Plot
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax2 = ax1.twinx()
    ax1.plot(steps, thresholds, color="red",
             label=r"$\sigma_i$", linewidth=3)
    ax2.plot(
        steps,
        n_testables,
        color="blue",
        label=r"$|\kappa_i(\sigma_i)|$",
        linewidth=3,
    )

    # Axis
    ax1.set_xlabel(r"Step $i$", fontsize=15)
    ax1.set_ylabel(r"Threshold $\sigma_i$", fontsize=15)
    ax2.set_ylabel(r"$|\kappa_i(\sigma_i)|$", fontsize=15)

    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax1.tick_params(axis="both", labelsize=13)
    ax2.tick_params(axis="both", labelsize=13)

    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(
        h1 + h2,
        l1 + l2,
        loc="lower right",
        fancybox="True",
        framealpha=0.9,
        fontsize=14,
    )
    fig.subplots_adjust(bottom=0.15, right=0.88)
    plt.savefig(f"./output/plot/threshold-{task}.png", bbox_inch="tight")
    plt.show()


def less_equal_useful(A, B):
    """
    Return 1 if A is equaly or less useful than B
    Keyword Arguments:
    A -- pattern A
    B -- pattern B
    """
    # Equal
    if A == B:
        return True

    A = np.array(A)
    B = np.array(B)

    # Uncomparable
    for i in range(len(A)):
        if i not in utility_feats and A[i] != B[i]:
            return False

    # More useful
    for i in utility_feats:
        if A[i] < B[i]:
            return False

    # Less useful
    return True


def groupping_by_min_p(patterns):
    """
    Helper function that groupping patterns by minimal p-values.
    Keyword Arguments:
    patterns -- list of patterns' data
    """
    patterns = sorted(patterns, key=lambda x: x["min_p"])
    min_p = -np.inf

    gr_patterns = OrderedDict()
    for pattern in patterns:
        if pattern["min_p"] > min_p:
            min_p = pattern["min_p"]
            gr_patterns[min_p] = [pattern]
        else:
            gr_patterns[min_p].append(pattern)

    return gr_patterns


def tarone_bonf(gr_patterns, alpha):
    """
    Discovering significant gr_patterns with Tarone-Bonferroni
    using LAMP implementation
    Keyword Arguments:
    gr_patterns -- List of gr_patterns's pattern
    alpha    -- significant level
    """

    print("\n-- Tarone-Bonferroni --")

    # Get avaible min p-values and number of gr_patterns for each min p-values
    min_ps = list(gr_patterns.keys())
    min_p_counts = [len(pattern) for pattern in gr_patterns.values()]

    # Find largest (min-p) pivot that can controls FWER
    i = 0
    pivot = min_ps[0]
    count = min_p_counts[0]
    prev_count = count
    while count * pivot <= alpha:
        prev_count = count
        if i + 1 < len(min_ps):
            i += 1
            count += min_p_counts[i]
            pivot = min_ps[i]
        else:
            break
    if prev_count * pivot > alpha:
        threshold = alpha / prev_count
    else:
        threshold = alpha / count

    # Reject
    rejects = []
    for patterns in gr_patterns.values():
        for pattern in patterns:
            if pattern["p"] <= threshold:
                rejects.append(pattern)
    rejects = sorted(rejects, key=lambda x: x["p"])
    return rejects


def remove_less_useful(gr_patterns, a):
    """
    Keyword Arguments:
    patterns         -- List of patterns' pattern
    a                -- pattern a that will be compared
    """
    new_gr_patterns = OrderedDict()
    for min_p, patterns in gr_patterns.items():
        new_patterns = []
        for pattern in patterns:
            if pattern["items"] != a["items"] and \
               not less_equal_useful(pattern["items"], a["items"]):
                new_patterns.append(pattern)
        if len(new_patterns) > 0:
            new_gr_patterns[min_p] = new_patterns
    return new_gr_patterns


def spur(gr_patterns, alpha):
    """
    Keyword Arguments:
    gr_patterns -- List of gr_patterns's pattern
    alpha       -- significant level
    """

    print("\n-- SPUR --")

    last_p = 0  # p-value of the last discovered pattern
    bucket = alpha  # significant budget
    rejects = []  # discovered patterns
    trace = []  # for post-analysis purpose
    it = 1
    last_threshold = 0
    while len(gr_patterns) > 0:
        # Get avaible minimal p-values
        min_ps = list(gr_patterns.keys())
        min_p_counts = [len(pattern) for pattern in gr_patterns.values()]

        # Find largest (min-p) threshold that can controls FWER
        # with similar strategy with LAMP
        i = 0
        min_p = min_ps[0]
        count = min_p_counts[0]
        prev_count = 0

        while count * (min_p - last_p) <= bucket:
            prev_count = count
            if i + 1 < len(min_ps):
                i += 1
                count += min_p_counts[i]
                min_p = min_ps[i]
            else:
                break

        if prev_count * (min_p - last_p) >= bucket:
            threshold = bucket / prev_count + last_p
        else:
            threshold = bucket / count + last_p

        # However, the strategy of LAMP sometimes return threshold that
        # smaller than last threshold, i.e., not optimal.
        # In that case, we use the threshold of the last iteration
        if (threshold < last_threshold):
            # Check if it satisfy the constraint
            assert (last_threshold - last_p) * prev_count <= bucket
            threshold = last_threshold

        n_testable = bucket / (threshold - last_p)
        trace.append((it, min_p, threshold, int(n_testable)))

        reject_p_value = np.inf
        reject_pattern = None
        for min_p, patterns in gr_patterns.items():
            for pattern in patterns:
                if pattern["p"] < reject_p_value:
                    reject_p_value = pattern["p"]
                    reject_pattern = pattern

        if reject_p_value <= threshold:
            rejects.append(reject_pattern)
            bucket -= (reject_p_value - last_p) * n_testable - reject_p_value
            last_p = reject_p_value
            gr_patterns = remove_less_useful(gr_patterns, reject_pattern)
        else:
            break

        it += 1
        last_threshold = threshold

    rejects = sorted(rejects, key=lambda x: x["p"])

    return rejects, trace


def usefulness_measure(A, B):
    """
    Usefulness measure from set A to B
    Keyword Arguments:
    A -- Rejected pattern set A
    B -- Rejected pattern set B
    """
    measure = 0
    for a in A:
        # Check if pattern 'a' is uncomparable/more-useful
        # than any patterns in B
        more_useful = 1
        for b in B:
            if less_equal_useful(a["items"], b["items"]):
                more_useful = 0
                break
        measure += more_useful
        if more_useful == 1:
            print(a["items"])

    return measure


def get_dominate_set(A):
    dominate_set = []
    for a in A:
        # Check if pattern 'a' is uncomparable/more-useful
        # than any other patterns in A
        dominate = 1
        for b in A:
            if a["items"] != b["items"] \
               and less_equal_useful(a["items"], b["items"]):
                dominate = 0
                break

        # Add to dominate set if pattern 'a' is a dominate pattern
        if dominate == 1:
            dominate_set.append(a["items"])

    dominate_set = sorted(dominate_set)
    return dominate_set


def print_pattern_set(A):
    """
    Keyword Arguments:
    A -- Pattern set A
    """
    for item in A:
        print(item)
    print(len(A))


def main():
    alpha = 0.05
    global utility_feats

    if len(sys.argv) > 1:
        task = sys.argv[1]
    else:
        print('python real_data_exp.py task_name alpha \n task_name in [adult, crash, property, society, property]')
        return 1

    if len(sys.argv) > 2:
        alpha = float(sys.argv[2])
    else:
        alpha = 0.05

    print(f"\n\n==== {task} ====")

    # Prepare data
    with open(f'data/preprocessed/{task}.pkl', "rb") as f:
        data_dict = pickle.load(f)
        patterns = data_dict["datas"]
        utility_feats = data_dict["utility_feats"]
        patterns = groupping_by_min_p(patterns)

    # Methods
    # == Tarone-Bonferroni ==
    tbonf_reject = tarone_bonf(patterns, alpha)
    print(f"Rejected {len(tbonf_reject)} hypotheses!")
    print_pattern_set(get_dominate_set(tbonf_reject))

    # == SPUR ==
    spur_reject, spur_trace = spur(patterns, alpha)
    print(f"Rejected {len(spur_reject)} hypotheses!")
    print_pattern_set(get_dominate_set(spur_reject))

    # == Compare SPUR and LAMP ==
    print("\n-- Distance --")
    print("d(LAMP, SPUR) =", usefulness_measure(tbonf_reject, spur_reject))
    print("d(SPUR, LAMP) =", usefulness_measure(spur_reject, tbonf_reject))

    # == Plot the threshold and testable number
    # plot_threshold(task, spur_trace)


if __name__ == "__main__":
    main()
