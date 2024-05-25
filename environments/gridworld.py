import typing

import numpy as np


class GridWorld:
    def __init__(
        self, rows: int, columns: int, max_iterations: typing.Optional[int] = 3
    ) -> None:
        self.max_row = rows - 1
        self.max_column = columns - 1
        self.values = np.zeros((rows, columns))
        self.terminal_states = [(0, 0), (self.max_row, self.max_column)]
        self.iterations = 0
        self.max_iterations = max_iterations
        self.valid_actions = ["up", "right", "down", "left"]

    def take_actions_for_a_state(self, row: int, column: int) -> float:
        state_value = 0
        for action in self.valid_actions:
            if action == "up":
                next_state = (max(row - 1, 0), column)
            elif action == "right":
                next_state = (row, min(column + 1, self.max_column))
            elif action == "down":
                next_state = (min(row + 1, self.max_row), column)
            else:
                next_state = (row, max(column - 1, 0))
            state_value += self.get_state_action_value(next_state)
        return state_value

    def get_state_action_value(self, next_state: typing.Tuple[int]) -> float:
        return 0.25 * (-1 + self.values[next_state])

    def iterate_states(self) -> None:
        for _ in range(self.max_iterations):
            next_values = np.zeros((self.max_row + 1, self.max_column + 1))
            for row in range(self.max_row + 1):
                for col in range(self.max_column + 1):
                    if (row, col) in self.terminal_states:
                        continue
                    next_values[(row, col)] = self.take_actions_for_a_state(row, col)
            self.values = next_values
