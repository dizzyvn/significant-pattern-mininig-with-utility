import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from synthesis_exp import methods, exps, sub_dir
from func.func import open_file, str_to_array


def read_result(f):
    """
    Read the result from files f
    Return the indexes of false and reject_list hypotheses
    """
    lines = f.readlines()
    # Indexes of false null hypotheses
    false_idxs = str_to_array(lines[0])
    # Indexes of reject_list hypotheses
    # Each line contains (possibly multiple) rejections for each run
    reject_list = [str_to_array(line) for line in lines[1:]]
    return false_idxs, reject_list


def type1_occured(false_idxs, reject_idxs):
    """
    Check if there is an type-I error occured
    i.e., if exists a rejection which is not a false null hypothesis
    """
    return int(len(np.union1d(false_idxs, reject_idxs)) > len(false_idxs))


def fwer(false_idxs, reject_list):
    """
    Given the indexes of false null hypotheses
    and the indexes of rejected hypotheses of multiple runs
    Return the FWER
    """
    n_hypotheses = len(reject_list)
    type1_error = 0
    for r in reject_list:
        type1_error += type1_occured(false_idxs, r)
    return type1_error / n_hypotheses


def count_rejection(false_idxs, reject_list):
    """
    Count the number of rejections for no-type-I-error runs.
    """
    counts = []
    for r in reject_list:
        n_rejects = len(r)
        if not type1_occured(false_idxs, r) and n_rejects > 0:
            counts.append(n_rejects)
    return counts


def ranks(false_idxs, rejected):
    """
    Return the maximal utility rank for each run
    """
    n_false_null = len(false_idxs)
    ranks = []
    for r in rejected:
        # We only concern runs that no type-I error occurs
        if not type1_occured(false_idxs, r) and len(r) > 0:
            optimal_idx = np.where(np.max(r) == false_idxs)[0][0]
            rank = n_false_null - optimal_idx
            ranks.append(rank)
    return np.array(ranks)


if __name__ == "__main__":
    # ===== Analyzing the result =====
    _fwer = dict()
    _rank = dict()
    _count = dict()
    for exp in exps:
        _fwer[exp] = dict()
        _rank[exp] = dict()
        _count[exp] = dict()
        for method in methods:
            print(f"\n==============\n{exp}, {method}")
            # Read result file
            f = open_file(sub_dir, exp, method, "r")
            false_idxs, reject_list = read_result(f)

            # Counting and stuff
            _fwer[exp][method] = fwer(false_idxs, reject_list)
            _rank[exp][method] = ranks(false_idxs, reject_list)
            _count[exp][method] = np.mean(count_rejection(false_idxs, reject_list))

            # Print result
            print("- FWER:", _fwer[exp][method])
            print("- RANKS:", np.histogram(_rank[exp][method])[0])
            print("- AVE-#REJECT:", _count[exp][method])

            # Close file
            f.close()

    # ===== Summarizing result =====
    print("\n====== SUMMARY ======")
    fwer_df = pd.DataFrame.from_dict(_fwer)
    count_df = pd.DataFrame.from_dict(_count)
    print(fwer_df)
    print(count_df)

    # ===== Plotting result =====
    # We don't plot invalid-spur
    _methods = [m for m in methods.keys() if m != "Invalid SPUR"]
    print(_methods)
    n_methods = len(_methods)
    n_exps = len(exps)

    linestyles = ["-", "--", ":"]
    colors = ["b", "g", "r"]
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    axs = axs.flat
    lines = []
    labels = []
    for exd, exp in enumerate(exps):
        axs[exd].set_title(
            exp, fontdict={"fontsize": 13},
        )
        for med, method in enumerate(_methods):
            hist = np.histogram(_rank[exp][method], bins=20)[0]
            ranks = np.arange(len(hist))

            # Plot histogram
            plot_id = exd
            line = axs[plot_id].plot(
                ranks,
                np.cumsum(hist),
                label=method,
                linewidth=3,
                linestyle=linestyles[med],
                color=colors[med],
            )
            # Settings
            axs[plot_id].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            axs[plot_id].set_ylim(bottom=0)
            xticks = np.array([0, 4, 9, 14, 19])
            xticklabels = xticks + 1
            axs[plot_id].set_xticks(xticks)
            axs[plot_id].set_xticklabels(xticklabels)
            axs[plot_id].set_xlabel("Utility ranking", fontsize=12)

            # Legends
            if exd == 0:
                lines.append(line)
                labels.append(method)

    axs[0].set_ylabel("Number of rejections", fontsize=12)
    axs[0].set_ylabel("Cumulative counts", fontsize=12)
    fig.legend(
        labels=labels, loc="lower center", ncol=3, borderaxespad=0.4, fontsize=12
    )
    fig.subplots_adjust(bottom=0.3)
    plt.savefig(f"output/plot/exp_1.png", bbox_inches="tight")
    plt.show()

    # ===== Printing result for latex's table format =====
    for method in methods:
        s = f"{method} "
        for exp in exps:
            s += "& %.3f " % _fwer[exp][method]
        for exp in exps:
            s += "& %.3f " % _count[exp][method]
        s += "\\\\"
        print(s)
