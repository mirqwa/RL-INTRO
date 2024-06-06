import collections
import json
import math
import os
import sys

import numpy as np
import pandas as pd

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import random_walk


def get_approximate_value(state: int, weights: np.array) -> float:
    state_group = math.ceil(state / 100) - 1
    state_representation = np.array([1 if i == state_group else 0 for i in range(10)])
    return state_representation, np.dot(state_representation, weights)


def get_target_values() -> dict:
    with open("monte_carlo/random_walk_state_values.json") as f:
        target_values = json.load(f)
        return target_values


def evaluate_policy() -> None:
    env = random_walk.RandomWalk(1000, 500, 100)
    alpha = 2 * 10**-5
    weights = np.zeros(10)
    target_values = get_target_values()
    state_counts = collections.defaultdict(int)
    for _ in range(100000):
        states, _ = env.generate_episode()
        for state in states[:-1]:
            state_counts[state] += 1
            target_value = target_values[str(state)]["average"]
            partial_derivatives, approximate_value = get_approximate_value(
                state, weights
            )
            weights += alpha * (target_value - approximate_value) * partial_derivatives
    print(weights)
    print(dict(state_counts))


if __name__ == "__main__":
    evaluate_policy()
