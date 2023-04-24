import numpy as np
import sys

sys.path.append("./agents/")
from utils import modify, figure_to_3drelief
from nlp_model.agent import DefArgs, init_models, predict_voxel


def target_to_subtasks(figure):
    zh, xh, yh = figure.hole_indx
    xy_holes = np.asarray(list(zip(xh, yh)))
    targets_plane = figure.relief.astype(int)
    color_plane = figure.figure_parametrs['color']
    X, Y = np.where(figure.relief != 0)
    addtional_tower_remote = (2, 2)
    # generate main blocks
    for x, y in zip(X, Y):
        for z in range(targets_plane[x, y]):
            custom_grid = np.zeros((9, 11, 11))
            if (color_plane is None) or (color_plane[z, x, y] == 0):
                custom_grid[z, x, y] = 1
                yield (x - 5, z - 1, y - 5, 1), custom_grid
            else:
                custom_grid[z, x, y] = int(color_plane[z, x, y])
                yield (x - 5, z - 1, y - 5, int(color_plane[z, x, y])), custom_grid
        if len(xy_holes) > 0 and x < (11 - addtional_tower_remote[0]) and y < (11 - addtional_tower_remote[1]):
            holes_in_xy = ((xy_holes - [x, y])[:, 0] == 0) & ((xy_holes - [x, y])[:, 1] == 0)
            holes_in_xy = np.where(holes_in_xy == 1)[0]
            additional_blocks = []
            last_height = 0
            z = 0
            # generate additional blocks
            #  print("Holes^")
            # print(zh[holes_in_xy])
            for height in zh[holes_in_xy]:
                for z in range(last_height, height):
                    custom_grid = np.zeros((9, 11, 11))
                    custom_grid[z, x + addtional_tower_remote[0], y + addtional_tower_remote[1]] = 1
                    additional_blocks.append((z, x + addtional_tower_remote[0], y + addtional_tower_remote[1]))
                    yield (x - 5 + addtional_tower_remote[0], z - 1, y - 5 + addtional_tower_remote[1], 1), custom_grid
                custom_grid = np.zeros((9, 11, 11))
                custom_grid[height, x, y] = -1
                last_height = height
                # make window
                yield (x - 5, height - 1, y - 5, -1), custom_grid
            # remove additional blocks
            if len(additional_blocks) > 0:
                for z, x, y in additional_blocks[::-1]:
                    custom_grid = np.zeros((9, 11, 11))
                    custom_grid[z, x, y] = -1
                    yield (x - 5, z - 1, y - 5, -1), custom_grid


def generate_preobs(min_value, max_value, red_degree=5):
    choice_range = list(range(min_value, int(max_value)))
    p = 1 / len(choice_range)
    p_for_bottom_block = p / red_degree
    addition_p = (p - p_for_bottom_block) / (len(choice_range) - 1)
    p += addition_p
    probs = [p_for_bottom_block] + [p] * (len(choice_range) - 1)
    return choice_range, probs


class Figure():
    def __init__(self, figure=None):
        self.use_color = True  # use color in figure generation
        self.figure = None  # figure without color
        self.figure_parametrs = None
        self.hole_indx = None  # all holes indexes
        self.simpl_holes = None  # holes only on the bottom
        self.relief = None  # 2d array of figure
        if figure:
            self.to_multitask_format(figure)

    def to_multitask_format(self, figure_with_colors):
        figure = np.zeros_like(figure_with_colors)
        figure[figure_with_colors > 0] = 1
        self.figure = figure.copy()
        _, _, full_figure = self.simplify()
        full_figure[full_figure != 0] = 1
        blocks = np.where((full_figure - self.figure) != 0)
        ind = np.lexsort((blocks[0], blocks[2], blocks[1]))
        self.hole_indx = (blocks[0][ind], blocks[1][ind], blocks[2][ind])
        figure_parametrs = {'figure': self.figure, 'color_': figure_with_colors}
        self.figure_parametrs = figure_parametrs
        return figure

    def simplify(self):
        if self.figure is not None:
            fig = self.figure.copy()
            is_modified, new_figure = modify(fig)
            target, relief = figure_to_3drelief(self.figure)

            relief = relief.max(axis=0)
            ones = np.ones((11, 11)) * np.arange(1, 10).reshape(-1, 1, 1)

            ones[ones <= relief] = 1
            ones[:, relief == 0] = 0
            ones[ones > relief] = 0
            full_figure = ones.copy()

            holes = relief - target.sum(axis=0)
            self.relief = relief
            self.simpl_holes = holes
        else:
            raise Exception("The figure is not initialized! Use 'make_task' method to do it!")
        return relief, holes, full_figure


class DialogueFigure(Figure):
    def __init__(self, *args):
        super().__init__(*args)
        self.args = DefArgs()
        configs = init_models(self.args)
        self.model, self.tokenizer, self.history, self.stats, self.voxel = configs
        self.full_voxel = None

    def clear_history(self):
        self.history = []
        self.voxel = np.zeros((11, 9, 11))
        return

    def right_color(self, figure):
        nlp_color = [-1, "red", "orange", "yellow", "green", "blue", "purple"]
        right_color = {
            'blue': 1,  # blue
            'green': 2,  # green
            'red': 3,  # red
            'orange': 4,  # orange
            'purple': 5,  # purple
            'yellow': 6,  # yellow
        }
        true_colors = [right_color[nlp_color[i]] for i in range(1, 7)]
        figure[figure == 1], figure[figure == 2], figure[figure == 3], \
        figure[figure == 4], figure[figure == 5], figure[figure == 6] = true_colors
        return figure

    def load_figure(self, right_voxel):
        right_voxel_bin = np.zeros_like(right_voxel)
        right_voxel_bin[right_voxel > 0] = 1

        figure = self.to_multitask_format(right_voxel_bin)
        # self.figure_parametrs['name'] = dialogue
        self.figure_parametrs['original'] = right_voxel
        self.figure_parametrs['color'] = right_voxel
        self.figure_parametrs['relief'] = self.relief
        return figure
