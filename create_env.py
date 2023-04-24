import gym
import numpy as np

from gym.spaces import Box
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper

from wrappers.common_wrappers import VisualObservationWrapper, JumpAfterPlace, EndActionController, ColorWrapper
from wrappers.loggers import SuccessRateFullFigure
from wrappers.multitask import TargetGenerator, SubtaskGenerator
from wrappers.reward_wrappers import RangetRewardFilledField, Closeness
from agents.mhb_baseline.generator import DialogueFigure


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        if all(dones):
            observations = self.env.reset()
        return observations, rewards, dones, infos


class FakeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space
        self.observation_space['obs'] = Box(0.0, 1.0, shape=(1,))

    def observation(self, observation):
        observation['obs'] = np.array([0.0])
        return observation


class InitializationCreator(gym.Wrapper):
    """This class fixes the GLU context bug during SampleFactory training"""

    def __init__(self, fake_for_first_n_resets=1, select_and_place=True, discretize=True, max_steps=1000):
        self._number_of_resets = 0
        self._fake_for_first_n_resets = fake_for_first_n_resets
        self._select_and_place = select_and_place
        self._discretize = discretize
        self._max_steps = max_steps

        self.env = self.create_env()
        super().__init__(self.env)

    def create_env(self):
        from gridworld.env import GridWorld
        from gridworld.tasks.task import Task

        fake = self._number_of_resets <= self._fake_for_first_n_resets
        env = GridWorld(render=True, select_and_place=self._select_and_place,
                        discretize=self._discretize,
                        max_steps=self._max_steps, fake=fake)
        custom_grid = np.ones((9, 11, 11))
        env.set_task(Task("", custom_grid, invariant=False))
        return env

    def reset(self):
        self.env = self.create_env()
        self._number_of_resets += 1

        return self.env.reset()


def make_iglu():
    env = InitializationCreator()
    env = TargetGenerator(env, fig_generator=DialogueFigure)
    env = SubtaskGenerator(env)
    env = VisualObservationWrapper(env)

    env = JumpAfterPlace(env)
    env = ColorWrapper(env)
    env = RangetRewardFilledField(env)
    env = Closeness(env)

    env = EndActionController(env)
    env = SuccessRateFullFigure(env)
    env = MultiAgentWrapper(env)
    env = AutoResetWrapper(env)

    return env
