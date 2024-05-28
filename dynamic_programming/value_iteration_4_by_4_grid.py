import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

import constants
from environments import gridworld
import utils


def value_iteration() -> None:
    env = gridworld.GridWorld(
        4, 4, constants.EQUIPROBABLE_POLICY, values_init_strategy="zeros"
    )
    utils.plot_policy(env, "The initial policy")
    theta = 0.0001
    num_of_iterations = 0
    while True:
        num_of_iterations += 1
        current_values = np.copy(env.values)
        env.update_state_values_to_maximum_action_values()
        diffs = current_values - env.values
        delta = np.abs(diffs).max()
        if delta < theta:
            break
    env.update_policy()
    print(f"Value iteration completed after {num_of_iterations} steps")
    utils.plot_policy(env, "The optimum policy")


if __name__ == "__main__":
    value_iteration()
