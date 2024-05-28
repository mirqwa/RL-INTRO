import os
import sys

from tabulate import tabulate

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import gridworld


action_symbols = {"left": "←", "right": "→", "up": "↑", "down": "↓"}


def plot_policy(env: gridworld.GridWorld, header: str) -> None:
    policy = []
    for row in range(env.max_row + 1):
        row_policy = [""] if row == 0 else [row - 1]
        for column in range(env.max_column + 1):
            if (row, column) in env.policy:
                actions = [
                    action_symbols[action]
                    for action, prob in env.policy[(row, column)].items()
                    if prob > 0
                ]
                actions = "".join(actions)
                row_policy.append(actions)
            else:
                row_policy.append("")
        policy.append(row_policy)
    print(header)
    print(tabulate(policy, headers=["", 0, 1, 2, 3]))
