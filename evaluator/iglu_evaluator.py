import uuid
from evaluator.utils import convert_keys_to_string

import numpy as np
import time

import os
import cv2
import json

class TaskEpisodeTracker:
    """
    Tracks episode results
    Since there are parallel envs
    The episode started first should be counted for evalution
    Each task has its own episode counter
    """

    def __init__(self, num_parallel_envs, max_episodes_per_task, rewards_file):
        self.num_parallel_envs = num_parallel_envs
        self.max_episodes_per_task = max_episodes_per_task
        self.episode_data = {}
        self.metrics_trackers = {}
        self.instance_id_to_uuid_map = {}
        self.uuid_task_map = {}
        self.rewards_file = rewards_file
        self.num_episodes_completed = 0

    def get_task_key_for_instance(self, instance_id):
        return self.uuid_task_map[self.instance_id_to_uuid_map[instance_id]]

    def register_reset(self, task_key, instance_id, task, first_obs):
        self.episode_data.setdefault(task_key, {})
        if len(self.episode_data[task_key]) < self.max_episodes_per_task:
            episode_uuid = str(uuid.uuid4())
            self.instance_id_to_uuid_map[instance_id] = episode_uuid
            self.uuid_task_map[episode_uuid] = task_key
            self.episode_data[task_key][episode_uuid] = {"completed": False}
            self.metrics_trackers[episode_uuid] = IGLUMetricsTracker(
                task_key, task, first_obs)

    def step(self, instance_id, observation, reward, info, action):
        episode_uuid = self.instance_id_to_uuid_map[instance_id]
        self.metrics_trackers[episode_uuid].step(observation, reward, info, action)

    def task_episodes_staged(self, task_key):
        return len(self.episode_data[task_key]) >= self.max_episodes_per_task

    def all_episodes_completed(self, num_total_tasks):
        # Ugly mess of if/for used here for efficiency
        if len(self.episode_data) < num_total_tasks:
            return False
        for task in self.episode_data:
            if not self.task_episodes_staged(task):
                return False
            for episode_id in self.episode_data[task]:
                if not self.episode_data[task][episode_id]["completed"]:
                    return False

        return True

    def add_metrics(self, instance_id, final_obs, user_termination):
        episode_uuid = self.instance_id_to_uuid_map[instance_id]
        task_key = self.uuid_task_map[episode_uuid]
        metrics = self.metrics_trackers[episode_uuid].get_metrics(final_obs)
        self.episode_data[task_key][episode_uuid]["metrics"] = metrics
        self.episode_data[task_key][episode_uuid]["user_termination"] = bool(user_termination)
        self.episode_data[task_key][episode_uuid]["completed"] = True
        self.num_episodes_completed += 1

        self.metrics_trackers[episode_uuid].dump_video(folder_path='./evaluator/videos', 
                                                       name=episode_uuid)

        del self.metrics_trackers[episode_uuid]

    def write_metrics_to_disk(self):
        stats = convert_keys_to_string(self.episode_data)
        with open(self.rewards_file, "w") as fp:
            json.dump(stats, fp)



from gridworld.tasks import Tasks, Task

class IGLUMetricsTracker:
    def __init__(self, task_key, subtask, first_obs):
        self.init = Tasks.to_dense(subtask.starting_grid)
        self.synthetic_task = Task(
            # create a synthetic task with only diff blocks. 
            # blocks to remove have negative ids.
            '', target_grid=subtask.target_grid - self.init
        )
        self.synthetic_target_grid_size = self.synthetic_task.target_size
        self.task_key = task_key
        self.total_reward = 0
        self.subtask = subtask
        self.stats = {}
        self.step_data = {"first_obs": first_obs,
                          "observations": [],
                          "rewards": [],
                          "infos": [],
                          "time": [],
                          "actions": []}
        self.episode_length = 1

    def step(self, observation, reward, info, action):
      #  print(action)
        self.step_data["observations"].append(observation)
        self.step_data["rewards"].append(reward)
        self.step_data["infos"].append(info)
        self.step_data["actions"].append(action)
        self.step_data["time"].append(time.time())
        self.total_reward += reward
        self.episode_length += 1

    def get_metrics(self, obs):
        argmax = self.synthetic_task.argmax_intersection(obs['grid'])
        synthetic_obs = obs['grid'] - self.init
        maximal_intersection = self.synthetic_task.get_intersection(synthetic_obs, *argmax)

        precision = maximal_intersection / (self.synthetic_target_grid_size + 1e-10)
        recall = maximal_intersection / (len(synthetic_obs.nonzero()[0]) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        self.stats['completion_rate_f1'] = f1
        self.stats['precision'] = precision
        self.stats['recall'] = recall
        self.stats['total_reward'] = self.total_reward
        self.stats['episode_length'] = self.episode_length

        return self.stats

    def dump_video(self, folder_path, name):
        name_key = '-'.join([str(s) for s in self.task_key])
        video_name = os.path.join(folder_path, name_key + '-' + str(name) + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        first_obs = self.step_data["first_obs"].get('pov', None) # if pov is present, we can render video
        if first_obs is None:
            return
        video_writer = cv2.VideoWriter(video_name, fourcc, 30, first_obs.shape[:2])
        video_writer.write(cv2.cvtColor(first_obs, cv2.COLOR_RGB2BGR))
        for obs in self.step_data["observations"]:
            video_writer.write(cv2.cvtColor(obs['pov'], cv2.COLOR_RGB2BGR))
        video_writer.release()