import sys
import json
import os
from os.path import join

import torch

from argparse import Namespace
from pathlib import Path
import numpy as np
import wandb
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES

from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint

sys.path.insert(1, "../")
from models import ResnetEncoderWithTarget
from config_validation import Experiment
from gridworld.env import GridWorld
from gridworld.tasks.task import Task
import gym
from gym.spaces import Box


class EndActionController(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.action_space.n + 1)


def make_iglu(*args, **kwargs):
    custom_grid = np.ones((9, 11, 11))
    env = GridWorld(render=False, select_and_place=True, discretize=True, max_steps=1000)
    env = EndActionController(env)
    env.set_task(Task("", custom_grid))
    return env


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='IGLUSilentBuilder-v0',
        make_env_func=make_iglu,
    )
    register_custom_encoder('custom_env_encoder', ResnetEncoderWithTarget)


# EXTRA_PER_POLICY_SUMMARIES.append(iglu_extra_summaries)

def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            full_config=exp.dict()
                            )
    return exp, flat_config


class APPOHolder:
    def __init__(self, algo_cfg):
        self.cfg = algo_cfg

        path = algo_cfg.path_to_weights
        device = algo_cfg.device
        register_custom_components()

        self.path = path
        self.env = None
        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        exp, flat_config = validate_config(config['full_config'])
        algo_cfg = flat_config

        env = create_env(algo_cfg.env, cfg=algo_cfg, env_config={})

        actor_critic = create_actor_critic(algo_cfg, env.observation_space, env.action_space)
        env.close()

        if device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        self.device = device

        # actor_critic.share_memory()
        actor_critic.model_to_device(device)
        policy_id = algo_cfg.policy_index
        checkpoints = join(path, f'checkpoint_p{policy_id}')
        checkpoints = LearnerWorker.get_checkpoints(checkpoints)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = algo_cfg

        self.rnn_states = None

    def after_reset(self, env):
        self.env = env

    @staticmethod
    def get_additional_info():
        return {"rl_used": 1.0}

    def act(self, observations, rewards=None, dones=None, infos=None):

        if self.rnn_states is None or len(self.rnn_states) != len(observations):
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)

        with torch.no_grad():
            #  print(self.rnn_states)
            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)
            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        return actions.cpu().numpy()

    def clear_hidden(self):
        self.rnn_states = None


def make_agent():
    register_custom_components()

    cfg = parse_args(argv=[
        '--algo=APPO', '--env=IGLUSilentBuilder-v0', '--experiment=rl_model',
        '--experiments_root=mhb_baseline',
        '--train_dir=./agents/'
    ],
        evaluation=True)
    cfg = load_from_checkpoint(cfg)

    cfg.setdefault("path_to_weights", "./agents/mhb_baseline/rl_model")
    return APPOHolder(cfg)
