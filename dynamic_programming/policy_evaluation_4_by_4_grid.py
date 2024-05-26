import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import gridworld

equiprobable_policypolicy = {
    (0, 1): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (0, 2): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (0, 3): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (1, 0): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (1, 1): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (1, 2): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (1, 3): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (2, 0): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (2, 1): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (2, 2): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (2, 3): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (3, 0): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (3, 1): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
    (3, 2): {"up": 0.25, "right": 0.25, "down": 0.25, "left": 0.25},
}


def evaluate_equiprobable_policy() -> None:
    env = gridworld.GridWorld(
        4, 4, equiprobable_policypolicy, values_init_strategy="zeros"
    )
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
    env = gridworld.GridWorld(
        4, 4, equiprobable_policypolicy, values_init_strategy="zeros"
    )
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
