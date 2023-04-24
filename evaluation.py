from agents.aicrowd_wrapper import AIcrowdAgent

import gym
import numpy as np
import gridworld
from gridworld.tasks import Tasks
from evaluator.iglu_evaluator import TaskEpisodeTracker

from evaluator.utils import convert_keys_to_string, read_json_file

import pdb
def create_single_env(action_space_name='walking'):
    return gym.make('IGLUGridworld-v0',
                    action_space=action_space_name,
                    size_reward=False, max_steps=500, vector_state=True)


class TaskGenerator:
    """
    Generator for IGLU tasks
    """

    def __init__(self, limit_tasks):
        from gridworld.data import IGLUDataset
        print("Loading tasks")
        self.dataset = IGLUDataset(dataset_version="v0.1.0-rc1",
                                   task_kwargs=None,
                                   force_download=False)
        self.datalen = len(self.dataset)

        self.sizes = {}
        for (task_id, session_id, subtask_id, subtask) in self.dataset:
            # Compute the difference between starting grid and target grid
            grid_difference = subtask.target_grid - Tasks.to_dense(subtask.starting_grid)
            # Compute the number of blocks which have to "change"
            size_of_non_zero_difference = len(grid_difference.nonzero()[0])
            # Save it in a dictionary, for future reference
            self.sizes[(task_id, session_id, subtask_id)] = size_of_non_zero_difference

        self.dataset_iter = iter(self.dataset)
        self.current_task_key = None
        self.current_task = None

        self.limit_tasks = limit_tasks  # Limit to first n subtasks instead of running through all tasks
        self.num_tasks_emitted = 0

    def __len__(self):
        return self.limit_tasks if self.limit_tasks is not None else self.datalen

    def get_next_task(self):
        if self.limit_tasks is not None and self.num_tasks_emitted > self.limit_tasks:
            return self.current_task_key, self.current_task
        self.num_tasks_emitted += 1

        next_task = next(self.dataset_iter, None)
        if next_task is not None:
            task_id, session_id, subtask_id, subtask = next_task
            self.current_task_key = task_id, session_id, subtask_id
            self.current_task = subtask
        return self.current_task_key, self.current_task


def evaluate(LocalEvalConfig):
    """
    Runs the local evaluation in same way as the aicrowd evaluator
    All episodes are run for the max number of episodes per task
    The first episodes that are started are tracked, any extra episodes per task are dropped
    """
    # Change at your own risk
    agent = AIcrowdAgent()
    num_parallel_envs = agent.num_parallel_envs
    action_space_name = agent.action_space_name

    task_generator = TaskGenerator(LocalEvalConfig.LIMIT_TASKS)

    episode_tracker = TaskEpisodeTracker(num_parallel_envs=num_parallel_envs,
                                         max_episodes_per_task=LocalEvalConfig.MAX_EPISODES_PER_TASK,
                                         rewards_file=LocalEvalConfig.REWARDS_FILE)

    ### Creating envs and running rollouts
    task_key, task = task_generator.get_next_task()
    envs = []
    current_tasks = []
    observations = []
    num_steps = 0
    for i in range(num_parallel_envs):
        env = create_single_env(action_space_name=action_space_name)
        pdb.set_trace()
        env.set_task(task)
        obs_res = env.reset()
        observations_agent = obs_res.copy()
        # These keys will not be provided by evaluator
        observations_agent.pop('agentPos', None)
        #  observations_agent.pop('grid', None)
        observations.append(observations_agent)
        envs.append(env)
        current_tasks.append(task)
        episode_tracker.register_reset(
            instance_id=i, task_key=task_key, task=task,
            first_obs=obs_res)
        if episode_tracker.task_episodes_staged(task_key):
            task_key, task = task_generator.get_next_task()

    rewards = [0.0] * num_parallel_envs
    dones = [False] * num_parallel_envs
    infos = [{}] * num_parallel_envs

    reset_data = observations, rewards, dones, infos

    actions, user_terminations = agent.register_reset(reset_data)
    while True:
        env_outputs = [env.step(action)
                       for env, action in zip(envs, actions)]
        observations, rewards, dones, infos = [], [], [], []

        for i, eo in enumerate(env_outputs):

            obs, rew, done, info = eo
            episode_tracker.step(i, obs, rew, info, actions[i])
            user_termination = user_terminations[i]
            if done or user_termination:
                # Save metrics for completed episode
                episode_tracker.add_metrics(instance_id=i,
                                            final_obs=obs,
                                            user_termination=user_termination)

                # Reset the environment
                task_key = episode_tracker.get_task_key_for_instance(
                    instance_id=i)
                if episode_tracker.task_episodes_staged(task_key):
                    task_key, task = task_generator.get_next_task()
                    envs[i].set_task(task)
                    current_tasks[i] = task
                obs = envs[i].reset()
                episode_tracker.register_reset(instance_id=i,
                                               task_key=task_key,
                                               task=current_tasks[i],
                                               first_obs=obs)

            observations_agent = obs.copy()
            # These keys will not be provided by the evaluator
            observations_agent.pop('agentPos', None)
            #  observations_agent.pop('grid', None)
            observations.append(observations_agent)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        step_data = observations, rewards, dones, infos

        actions, user_terminations = agent.compute_action(step_data)

        episodes_completed = episode_tracker.num_episodes_completed

        # Episodes completed
        if episode_tracker.all_episodes_completed(num_total_tasks=len(task_generator)):
            print(f"Episodes completed: {episodes_completed}")
            break

        num_steps += 1
        if num_steps % 500 == 0:
            print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

    episode_tracker.write_metrics_to_disk()
    print("Rollout phase complete")

    ### Calculate scores
    all_episodes_data = read_json_file(LocalEvalConfig.REWARDS_FILE)

    task_summaries = {}
    for task_key_str, ep_data in all_episodes_data.items():
        ep_metrics = [v["metrics"] for k, v in ep_data.items() if v["completed"]]
        if len(ep_metrics) == 0:
            continue
        average_metrics = {}
        for k in ep_metrics[0]:
            average_metrics[k] = np.mean([epm[k] for epm in ep_metrics])
        task_summaries[task_key_str] = average_metrics

    sizes = convert_keys_to_string(task_generator.sizes)
    metrics_size_averaged = {}
    metric_keys = task_summaries[list(task_summaries.keys())[0]].keys()
    for metric in metric_keys:
        metrics_size_averaged[metric] = np.sum([avgm[metric] * sizes[k] for k, avgm in task_summaries.items()]) \
                                        / sum(sizes.values())

    print("===================== Final scores =======================")
    print(metrics_size_averaged)


if __name__ == "__main__":
    # change the local config as needed
    class LocalEvalConfig:
        MAX_EPISODES_PER_TASK = 2  # should be >= 2
        REWARDS_FILE = './evaluator/metrics.json'
        LIMIT_TASKS = None  # set this to none for all public tasks


    evaluate(LocalEvalConfig)
