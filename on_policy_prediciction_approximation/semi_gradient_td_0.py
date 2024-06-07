import collections
import json
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import random_walk


def get_approximate_value(state: int, weights: np.array) -> float:
    state_group = math.ceil(state / 100) - 1
    state_representation = np.array([1 if i == state_group else 0 for i in range(10)])
    return state_representation, np.dot(state_representation, weights)


def evaluate_policy() -> None:
    env = random_walk.RandomWalk(1000, 500, 100)
    alpha = 2 * 10**-5
    weights = np.zeros(10)
    current_state = 500
    state_counts = collections.defaultdict(int)
    for _ in range(10000):
        state_counts[current_state] += 1
        next_state, reward = env.take_a_step()
        partial_derivatives, approximate_value = get_approximate_value(
            current_state, weights
        )
        _, next_state_value = get_approximate_value(next_state, weights)
        weights += (
            alpha
            * (reward + next_state_value - approximate_value)
            * partial_derivatives
        )
        current_state = next_state
    print(weights)


if __name__ == "__main__":
    evaluate_policy()
