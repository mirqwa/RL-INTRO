import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def get_approximate_value(state: int, weights: np.array) -> float:
    state_group = math.ceil(state / 100) - 1
    state_representation = np.array([1 if i == state_group else 0 for i in range(10)])
    return state_representation, np.dot(state_representation, weights)


def plot_values(
    true_values: dict, approximate_values: list, state_counts: dict, title: str
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
        title,
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
