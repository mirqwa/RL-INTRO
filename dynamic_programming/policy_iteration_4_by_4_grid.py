import copy
import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

import constants
from environments import gridworld
import utils


def evaluate_policy(env: gridworld.GridWorld) -> None:
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

def improve_policy(env: gridworld.GridWorld) -> bool:
    current_policy = copy.deepcopy(env.policy)
    env.update_policy()
    return current_policy == env.policy


def iterate_equiprobable_policy() -> None:
    env = gridworld.GridWorld(
        4,
        4,
        constants.EQUIPROBABLE_POLICY,
        values_init_strategy="zeros",
    )
    utils.plot_policy(env, "The initial policy")
    policy_iterations = 0
    while True:
        policy_iterations += 1
        print(f"Policy iteration {policy_iterations}")
        evaluate_policy(env)
        if improve_policy(env):
            break
    print(f"Policy iteration finished after {policy_iterations} steps")
    utils.plot_policy(env, "The optimum policy")

if __name__ == "__main__":
    iterate_equiprobable_policy()
