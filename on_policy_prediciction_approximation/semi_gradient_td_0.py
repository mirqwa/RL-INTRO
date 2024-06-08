import collections
import json
import math
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
    initial_alpha = 10**-2
    weights = np.zeros(10)
    state_counts = collections.defaultdict(int)
    iterations = 1000000
    discounting_factor = 1
    for i in range(iterations):
        initial_alpha *= (1 / (1 + 0.00001 * i))
        print(f"Running simulation number {i + 1}")
        current_weights = np.array(weights)
        env.reset_state()
        steps = 0
        avg_steps = 0
        while not env.terminated:
            decay_rate = 1 / avg_steps if avg_steps else 1 / 100
            alpha = initial_alpha * (1 / (1 + decay_rate * steps))
            current_state = env.current_state
            state_counts[current_state] += 1
            next_state, reward = env.take_a_step()
            partial_derivatives, approximate_value = utils.get_approximate_value(
                current_state, weights
            )
            _, next_state_value = utils.get_approximate_value(next_state, weights)
            weights += (
                alpha
                * (reward + discounting_factor * next_state_value - approximate_value)
                * partial_derivatives
            )
            steps += 1
            if env.terminated:
                avg_steps += (steps - avg_steps) / (steps)
        diffs = np.array(weights) - current_weights
        # print(np.abs(diffs).max())
        if np.abs(np.array(weights)).max() > 0.9:
            break
    print(weights)
    utils.plot_values(
        get_target_values(),
        weights,
        state_counts,
        "Function approx by state agg on 1000-state random-walk, using semi-gradient TD(0)",
    )

# [-0.90550739 -0.5020662  -0.28410896 -0.14275844 -0.03500132  0.05368199, 0.15405536  0.28908088  0.50397257  0.90248466]


if __name__ == "__main__":
    evaluate_policy()
