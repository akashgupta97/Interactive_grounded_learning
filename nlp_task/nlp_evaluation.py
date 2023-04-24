import os

import numpy as np

from agents.nlp_model.agent import GridPredictor
from agents.nlp_model.utils import plot_grid
from evaluator.iglu_evaluator import IGLUMetricsTracker


def compute_metric(grid, subtask):
    igm = IGLUMetricsTracker(None, subtask, {})
    return igm.get_metrics({'grid': grid})


def get_dialog(subtask):
    import gym
    env = gym.make('IGLUGridworld-v0')
    env.set_task(subtask)
    obs = env.reset()
    return obs['dialog']


def main():
    grid_predictor = None

    from gridworld.tasks import Task

    from gridworld.data import IGLUDataset
    dataset = IGLUDataset(task_kwargs=None, force_download=False, )

    total_score = []

    for j, (task_id, session_id, subtask_id, subtask) in enumerate(dataset):
        str_id = str(task_id) + '-session-' + str(session_id).zfill(3) + '-subtask-' + str(subtask_id).zfill(3)
        print('Starting task:', str_id)
        subtask: Task = subtask

        if grid_predictor is None:
            grid_predictor = GridPredictor()

        dialog = get_dialog(subtask)
        predicted_grid = grid_predictor.predict_grid(dialog)

        plots_dir = 'nlp-evaluation-plots'

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        f1_score = round(compute_metric(predicted_grid, subtask)['completion_rate_f1'], 3)
        results = {'F1': f1_score}
        total_score.append(f1_score)
        results_str = " ".join([f"{metric}: {value}" for metric, value in results.items()])
        plot_grid(predicted_grid, text=str_id + ' ' + f'({results_str})' + "\n" + dialog).savefig(
            f'./{plots_dir}/{str_id}-predicted.png')
        plot_grid(subtask.target_grid, text=str_id + " (Ground truth)\n" + dialog).savefig(
            f'./{plots_dir}/{str_id}-gt.png')

    print('Total F1 score:', np.mean(total_score))


if __name__ == '__main__':
    main()
