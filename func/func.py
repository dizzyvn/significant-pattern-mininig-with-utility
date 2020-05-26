import os
import numpy as np

base_dir = "/home/dizzy/Dropbox/tsukuba/labo/paper/useful-significant-pattern/code/"
log_dir = f"{base_dir}/log/"
fig_output_dir = f"{base_dir}/plot/"


def get_name(exp, method):
    exp = exp.replace(" ", "-").lower()
    method = method.replace(" ", "-").lower()

    return f"{exp}_{method}"


def open_file(sub_dir=None, exp="", method="", operation="w"):
    if sub_dir is not None:
        path = f"{log_dir}{sub_dir}"
    if not os.path.exists(path):
        os.makedirs(path)

    return open(f"{path}{get_name(exp, method)}.txt", operation)


def write_list(f, results):
    for x in results:
        f.write(f"{x} ")
    f.write("\n")


def str_to_array(s):
    a = [int(i) for i in s.split(" ")[:-1]]
    return np.array(a)
