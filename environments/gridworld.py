import typing

import numpy as np


class GridWorld:
    def __init__(
        self,
        rows: int,
        columns: int,
        policy: typing.Optional[str] = "equiprobable",
        max_iterations: typing.Optional[int] = 10000,
        theta: typing.Optional[float] = 0.001,
    ) -> None:
        self.max_row = rows - 1
        self.max_column = columns - 1
        self.values = np.zeros((rows, columns))
        self.terminal_states = [(0, 0), (self.max_row, self.max_column)]
        self.iterations = 0
        self.theta = theta
        self.max_iterations = max_iterations
        self.valid_actions = ["up", "right", "down", "left"]
        self.policy = policy

    def get_state_action_value(self, next_state: typing.Tuple[int]) -> float:
        return 0.25 * (-1 + self.values[next_state])

    def take_equiprobable_actions_for_a_state(self, row: int, column: int) -> float:
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

    def take_actions(self) -> np.array:
        next_values = np.zeros((self.max_row + 1, self.max_column + 1))
        for row in range(self.max_row + 1):
            for col in range(self.max_column + 1):
                if (row, col) in self.terminal_states:
                    continue
                if self.policy == "equiprobable":
                    next_values[
                        (row, col)
                    ] = self.take_equiprobable_actions_for_a_state(row, col)
                else:
                    raise NotImplementedError(f"{self.policy} not implemented")
        return next_values

    def iterate_states(self) -> None:
        for i in range(self.max_iterations):
            next_values = self.take_actions()
            diffs = next_values - self.values
            delta = np.abs(diffs).max()
            self.values = next_values
            if delta < self.theta:
                print(
                    f"Not expecting further improvement, stopping after {i + 1} iterations"
                )
                break
        self.values = np.round(self.values, 2)
