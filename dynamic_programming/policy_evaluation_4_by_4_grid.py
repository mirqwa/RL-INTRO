import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import gridworld


def evaluate_equiprobable_policy() -> None:
    env = gridworld.GridWorld(4, 4, values_init_strategy="zeros")
    theta = 0.0001
    num_of_iterations = 0
    while True:
        num_of_iterations += 1
        current_values = np.copy(env.values)
        env.values = env.take_actions()
        diffs = current_values - env.values
        delta = np.abs(diffs).max()
        if delta < theta:
            break
    print(f"Policy evaluation completed after {num_of_iterations} steps")
    print(np.round(env.values, 2))


def evaluate_equiprobable_policy_with_inplace_update() -> None:
    env = gridworld.GridWorld(4, 4, values_init_strategy="zeros")
    theta = 0.0001
    num_of_iterations = 0
    while True:
        num_of_iterations += 1
        current_values = np.copy(env.values)
        env.take_actions(inplace_update=True)
        diffs = current_values - env.values
        delta = np.abs(diffs).max()
        if delta < theta:
            break
    print(
        f"Policy evaluation with in-place update completed after {num_of_iterations} steps"
    )
    print(np.round(env.values, 2))


if __name__ == "__main__":
    evaluate_equiprobable_policy()
    evaluate_equiprobable_policy_with_inplace_update()
