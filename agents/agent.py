import sys
import json
import os
from os.path import join

import torch
import wandb
import re
import sys

sys.path.append("agents/mhb_baseline/")
from models.models import ResnetEncoderWithTarget
import matplotlib.pyplot as plt
from step_by_step_enjoy import APPOHolder, make_agent
from generator import DialogueFigure
from nlp_model.agent import DefArgs, init_models, predict_voxel, GridPredictor
from generator import DialogueFigure, target_to_subtasks
import cv2
import numpy as np


class MultitaskHierarchicalAgent:
    def __init__(self):
        self.grid_predictor = GridPredictor()

        self.actions_space = None
        self.target_grid = None
        self.steps = 0
        self.args = DefArgs()

        self.figure = DialogueFigure()
        self.subtasks = None
        self.move_action = [6, 7, 8, 9, 10, 11]
        self.jump_flag = 0
        self.action = None
        self.last_action = None
        self.log = False
        self.jump_count = 2
        self.obs_stack = [None, None]
        self.start = True
        self.last_was_action = False
        self.termination = False
        self.action_queue = []
        self.commands = None
        self.subtask_agent = make_agent()

        self.current_dialog = None

    def set_action_space(self, action_space):
        self.actions_space = action_space

    def act(self, observation, reward, done, info):
        if done:
            self.clear_variable(observation)

            # making empty action to safely go to next episode
            return 0, False

        if self.start:
            episode = np.random.randint(0, 100)
            self.out = cv2.VideoWriter(f'video/{episode}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                       20, (64, 64))
            dialog = self.dialogue_to_commands(observation['dialog'])
            self.commands = dialog
            predicted_grid = self.grid_predictor.predict_grid(observation['dialog'])
            self.figure.load_figure(predicted_grid)
            self.subtasks = target_to_subtasks(self.figure)
            self.target_grid, termation = self.try_update_task()
            self.start = False

        self.out.write(cv2.resize(observation['pov'][:, :, ::-1], (64, 64)))
        action_generation, action, termation = self.do_action_from_stack()
        if action_generation:

            self.update_obs_stack(observation)
            self.target_grid, termation = self.try_update_task()
            self.last_action = self.action
            observation_for_model = self.obs_for_model(observation)
            action = self.subtask_agent.act([observation_for_model])
            self.action = action[0]
            if self.action >= 17:
                action = 0
            if action in self.move_action:
                action = self.choose_right_color(action)

                jumps = [5 for _ in range(self.jump_count - 1)]
                self.put([action, *jumps])
                action = 5

        if termation:
            self.clear_variable(observation)
        return action, termation

    def clear_variable(self, observation):
        self.start = True
        self.subtask_agent.clear_hidden()
        self.figure.clear_history()
        self.commands = []
        self.out.release()
        self.last_action = None
        self.action = None
        pass

    def put(self, actions):
        self.action_queue = actions
        return

    def update_obs_stack(self, observation):
        kernel = np.ones((5, 5), np.float32) / 25
        self.obs_stack[-1], self.obs_stack[0] = self.obs_stack[0], self.obs_stack[-1]
        img = observation['pov']
        smooth_img = cv2.filter2D(img, -1, kernel)
        self.obs_stack[-1] = smooth_img
        return

    def put_success(self):
        v = np.random.random()
        if np.sum(self.target_grid[0]) == 0 and v > 0.3:
            return True
        if (self.obs_stack[0] is None) or (self.obs_stack[1] is None):
            return False
        diff = abs(self.obs_stack[-1].mean(axis=2) - self.obs_stack[0].mean(axis=2))
        num = np.random.randint(0, 100)
        xc, yc = np.array(diff.shape) // 2
        thresh = 60
        if diff[xc, yc] >= thresh:
            return True
        return False

    def try_update_task(self):
        if self.start or (self.last_action in self.move_action and self.action == 18):
            try:
                _, target_grid = next(self.subtasks)
                return target_grid, False
            except Exception as e:
                return self.target_grid, True
        return self.target_grid, False

    def dialogue_to_commands(self, full_dialogue):
        commands = re.split("(<Architect>)|(<Builder>)", full_dialogue)
        no_none = [command for command in commands if command is not None]
        no_zero_len = [command for command in no_none if len(command) > 0]
        atchitect = no_zero_len[1::4]
        return atchitect

    def obs_for_model(self, observation):
        obs = {'obs': observation['pov'],
               'compass': observation['compass'],
               'inventory': observation['inventory'],
               'target_grid': self.target_grid}
        return obs

    def do_action_from_stack(self):
        try:
            action = self.action_queue.pop()
            return False, action, False
        except:
            return True, None, False

    def choose_right_color(self, action):

        tcolor = np.sum(self.target_grid)
        colors_to_hotbar = {
            'blue': 1,  # blue
            'green': 2,  # green
            'red': 3,  # red
            'orange': 4,  # orange
            'purple': 5,  # purple
            'yellow': 6,  # yellow
        }

        colors = list(colors_to_hotbar.keys())
        idx = list(colors_to_hotbar.values())
        # hotbar_to_color = dict(zip(idx, colors))
        action = int(5 + tcolor)
        return action
