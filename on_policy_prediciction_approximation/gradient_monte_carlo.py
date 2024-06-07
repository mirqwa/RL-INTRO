import collections
import json
import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import random_walk
import utils


def get_target_values() -> dict:
    with open("monte_carlo/random_walk_state_values.json") as f:
        target_values = json.load(f)
        return {int(state): value["average"] for state, value in target_values.items()}


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
            target_value = target_values[state]
            partial_derivatives, approximate_value = utils.get_approximate_value(
                state, weights
            )
            weights += alpha * (target_value - approximate_value) * partial_derivatives
    utils.plot_values(
        target_values,
        weights,
        state_counts,
        "Function approx by state agg on 1000-state random-walk, using gradient Monte Carlo",
    )


if __name__ == "__main__":
    evaluate_policy()
