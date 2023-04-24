import re

import matplotlib.pyplot as plt
import numpy as np

def figure_to_3drelief(target_):
    target = np.zeros_like(target_)
    target[target_ != 0] = 1
    relief = target * np.arange(1, 10).reshape(-1, 1, 1)
    return target, relief


def modif_tower(a):
    idx = np.where(a == 1)[0]
    diff = idx[1:] - idx[:-1]
    holes = np.where(diff > 1)
    modif = False
    if len(holes[0]) > 0:
        ones = np.where(a == 1)[0][-1]
        a[idx[holes[0][0]] + 1:ones] = 1
        modif = True
    return a, modif


def modify(figure):
    modifs_list = []
    new_figure = np.zeros_like(figure)
    modifs = False
    for i in range(figure.shape[1]):
        for j in range(figure.shape[2]):
            tower = figure[:, i, j]
            tower[tower > 0] = 1
            new_figure[:, i, j], flag = modif_tower(tower)
            modifs |= flag
            binary = "".join(str(figure[:, i, j]).split(" "))
            binary = re.sub('[123456789]', '1', binary)
            p = re.findall("10+1", binary)
            if len(p) > 0:
                modifs_list.append([i, j])
    return modifs_list, new_figure


if __name__ == "__main__":
    pass
