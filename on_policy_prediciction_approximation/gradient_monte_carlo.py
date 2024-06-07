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


def get_target_values() -> dict:
    with open("monte_carlo/random_walk_state_values.json") as f:
        target_values = json.load(f)
        return {int(state): value["average"] for state, value in target_values.items()}


def plot_values(
    true_values: dict, approximate_values: list, state_counts: dict
) -> None:
    _, ax = plt.subplots()
    ax2 = ax.twinx()
    true_values_x = list(true_values.keys())[1:-1]
    true_values_y = list(true_values.values())[1:-1]
    ax.plot(true_values_x, true_values_y, c="red", label="True Values")
    ax.step(
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        approximate_values,
        c="blue",
        label="Approximate MC value",
    )

    state_counts = [key for key, val in state_counts.items() for _ in range(val)]
    ax2.hist(state_counts, bins=998, color="grey", label="State distribution")

    ax.set_title(
        "Function approx by state agg on 1000-state random-walk, using gradient Monte Carlo",
        fontsize=20,
        fontstyle="italic",
    )
    ax.set_xlabel("State", fontsize=18)
    ax.set_ylabel("Value scale", fontsize=18)
    ax2.set_ylabel("Distribution scale", fontsize=18)

    # add legend manually
    handles, _ = ax.get_legend_handles_labels()
    rectangle = Rectangle((0, 0), 1, 1, color="grey", label="State distribution")
    handles.append(rectangle)
    plt.legend(handles=handles, loc="upper left")

    plt.show()


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
            partial_derivatives, approximate_value = get_approximate_value(
                state, weights
            )
            weights += alpha * (target_value - approximate_value) * partial_derivatives
    plot_values(target_values, weights, state_counts)


if __name__ == "__main__":
    evaluate_policy()
