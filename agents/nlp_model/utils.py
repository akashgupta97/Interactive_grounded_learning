import time

import numpy as np
from gridworld.data import IGLUDataset
from gridworld.tasks import Tasks, Task
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from evaluator.iglu_evaluator import IGLUMetricsTracker


def parse_logs(logs, history_len=None):
    if history_len:
        logs = logs[-history_len:]
    input_line = ''
    for item in logs:
        input_line += f'<Architect> {item[0]} <Builder> {item[1]} <sep1> '
    return input_line


colours = {'red': 1, 'orange': 2, 'yellow': 3, 'green': 4, 'blue': 5, 'purple': 6}

x_orientation = ['right', 'left']
y_orientation = ['lower', 'upper']
z_orientation = ['before', 'after']


def logging(log_file, log):
    if not log_file:
        print(log)
    else:
        log_file.writelines(log + '\n')


def parse_coords(action, x_0=0, y_0=0, z_0=0):
    x_left_find = action.find(x_orientation[0])
    x_right_find = action.find(x_orientation[1])
    if x_right_find != -1:
        x_pos = x_0 + int(action[x_right_find - 3:x_right_find - 1])
    elif x_left_find != -1:
        x_pos = x_0 - int(action[x_left_find - 3:x_left_find - 1])
    else:
        x_pos = x_0

    y_left_find = action.find(y_orientation[0])
    y_right_find = action.find(y_orientation[1])
    if y_right_find != -1:
        y_pos = y_0 + int(action[y_right_find - 2])
    elif y_left_find != -1:
        y_pos = y_0 - int(action[y_left_find - 2])
    else:
        y_pos = y_0

    z_left_find = action.find(z_orientation[0])
    z_right_find = action.find(z_orientation[1])
    if z_right_find != -1:
        z_pos = z_0 + int(action[z_right_find - 3:z_right_find - 1])
    elif z_left_find != -1:
        z_pos = z_0 - int(action[z_left_find - 3:z_left_find - 1])
    else:
        z_pos = z_0

    return x_pos, y_pos, z_pos


def update_state_from_action(args, state, action, init_block, log_file=None):
    x_0, y_0, z_0 = init_block
    action = action.strip()
    command = action.split()[0]
    ok = True

    if not command in ['pick', 'put']:
        if args.verbose > 1:
            logging(log_file, f'action: {action}, Invalid command {command}')
        ok = False
        return state, ok

    if len(action.split()) <= 1:
        return state, False

    if action.split()[1] == 'initial':
        colour = action.split()[2]
        if not colour in colours.keys():
            if args.verbose > 1:
                logging(log_file, f'action: {action}, Invalid colour {colour}')
            ok = False
            return state, ok

        if command == 'put':
            if state[init_block] == 0:
                state[init_block] = colours[colour]
            else:
                if args.verbose > 1:
                    logging(log_file, f'action: {action}, Wrong action, position occupied')
                ok = False
                return state, ok
        else:
            if state[init_block] == colours[colour]:
                state[init_block] = 0
            else:
                if args.verbose > 1:
                    logging(log_file,
                            f'action: {action} Wrong action, colours dont match: {state[init_block]}, {colours[colour]}')
                ok = False
                return state, ok
    else:
        colour = action.split()[1]
        if not colour in colours.keys():
            if args.verbose > 1:
                logging(log_file, f'action: {action}, Invalid colour {colour}')
            ok = False
            return state, ok
        # print('------' * 20)
        # print(action)
        try:
            x, y, z = parse_coords(action, x_0, y_0, z_0)
        except ValueError:
            if args.verbose > 1:
                logging(log_file, f'action: {action}, Invalid coordinates')
            ok = False
            return state, ok
        if (x not in range(0, 11)) or (y not in range(0, 9)) or (z not in range(0, 11)):
            if args.verbose > 1:
                logging(log_file, f'action: {action}, Invalid coordinates {x}, {y}, {z}')
            ok = False
            return state, ok

        if command == 'put':
            if state[x, y, z] == 0:
                state[x, y, z] = colours[colour]
            else:
                if args.verbose > 1:
                    logging(log_file, f'action: {action}, Wrong action, position occupied')
                ok = False
                return state, ok

        else:
            if state[x, y, z] == colours[colour]:
                state[x, y, z] = 0
            else:
                if args.verbose > 1:
                    logging(log_file,
                            f'action: {action}, Wrong action, colours dont match: {state[x, y, z]}, {colours[colour]}')
                ok = False
                return state, ok

    return state, ok


def plot_voxel(voxel, text=None):
    idx2color = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange', 5: 'purple', 6: 'yellow'}
    vox = voxel.transpose(1, 2, 0)
    colors = np.empty(vox.shape, dtype=object)
    for i in range(vox.shape[0]):
        for j in range(vox.shape[1]):
            for k in range(vox.shape[2]):
                if vox[i, j, k] != 0:
                    colors[i][j][k] = str(idx2color[vox[i, j, k]])

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(projection='3d', )
    ax.voxels(vox, facecolors=colors, edgecolor='k', )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    ax.set_xticks(np.arange(0, 12, 1), minor=True)
    ax.set_yticks(np.arange(0, 12, 1), minor=True)
    ax.set_zticks(np.arange(0, 9, 1), minor=True)

    box = ax.get_position()
    box.x0 = box.x0 - 0.05
    box.x1 = box.x1 - 0.05
    box.y1 = box.y1 + 0.16
    box.y0 = box.y0 + 0.16
    ax.set_position(box)

    if text is not None:
        plt.annotate(text, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points',
                     verticalalignment='top', wrap=True)
    return fig


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_grid(voxel, text=None):
    idx2color = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange', 5: 'purple', 6: 'yellow'}
    vox = voxel.transpose(1, 2, 0)
    colors = np.empty(vox.shape, dtype=object)
    for i in range(vox.shape[0]):
        for j in range(vox.shape[1]):
            for k in range(vox.shape[2]):
                if vox[i, j, k] != 0:
                    colors[i][j][k] = str(idx2color[vox[i, j, k]])

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(projection='3d', )
    ax.voxels(vox, facecolors=colors, edgecolor='k', )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    ax.set_xticks(np.arange(0, 12, 1), minor=True)
    ax.set_yticks(np.arange(0, 12, 1), minor=True)
    ax.set_zticks(np.arange(0, 9, 1), minor=True)

    box = ax.get_position()
    box.x0 = box.x0 - 0.05
    box.x1 = box.x1 - 0.05
    box.y1 = box.y1 + 0.16
    box.y0 = box.y0 + 0.16
    ax.set_position(box)

    if text is not None:
        plt.annotate(text, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points',
                     verticalalignment='top', wrap=True)
    return fig


def compute_metric(grid, target_grid):
    sb = IGLUDataset(task_kwargs={'invariant': False}).reset()
    sb.target_grid = target_grid
    igm = IGLUMetricsTracker(None, sb, {'grid': grid})
    return igm.get_metrics({'grid': grid})


def main():
    example = np.zeros(shape=(9, 11, 11))
    blocks = [[0, 5, 5],
              # [1, 5, 6],
              # [2, 5, 6],
              # [3, 5, 5],
              # [4, 5, 6],
              # [3, 5, 6],
              # [3, 5, 7],
              ]
    for block in blocks:
        example[block[0], block[1], block[2]] = 6
    plot_grid(example,
              text="<Architect> Facing north, place a yellow block four spaces from the eastern edge. Place another yellow block on top of that one.").show()


if __name__ == '__main__':
    main()
