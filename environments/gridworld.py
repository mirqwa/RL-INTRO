import typing

import numpy as np


class GridWorld:
    def __init__(
        self,
        rows: int,
        columns: int,
        policy: dict,
        max_iterations: typing.Optional[int] = 10000,
        theta: typing.Optional[float] = 0.001,
        values_init_strategy: typing.Optional[str] = "zeros",
    ) -> None:
        self.max_row = rows - 1
        self.max_column = columns - 1
        self.values = self.initialize_values(values_init_strategy)
        self.terminal_states = [(0, 0), (self.max_row, self.max_column)]
        self.iterations = 0
        self.theta = theta
        self.max_iterations = max_iterations
        self.valid_actions = ["up", "right", "down", "left"]
        self.policy = policy

    def initialize_values(self, values_init_strategy: str) -> None:
        if values_init_strategy == "zeros":
            return np.zeros((self.max_row + 1, self.max_column + 1))
        raise NotImplementedError(
            f"Initialization values initialization strategy for {values_init_strategy} is not implemented"
        )

    def get_state_action_value(
        self, action_probability: float, next_state: typing.Tuple[int]
    ) -> float:
        return action_probability * (-1 + self.values[next_state])

    def get_action_value(
        self, row: int, column: int, action: str, action_probability: float
    ) -> float:
        if action not in self.valid_actions:
            raise ValueError(
                f"Invalid action: {action} not allowed in ({row}, {column})"
            )
        if action == "up":
            next_state = (max(row - 1, 0), column)
        elif action == "right":
            next_state = (row, min(column + 1, self.max_column))
        elif action == "down":
            next_state = (min(row + 1, self.max_row), column)
        else:
            next_state = (row, max(column - 1, 0))
        return self.get_state_action_value(action_probability, next_state)

    def take_actions_for_a_state(self, row: int, column: int) -> float:
        state_value = 0
        for action, action_probability in self.policy[(row, column)].items():
            state_value += self.get_action_value(
                row, column, action, action_probability
            )
        return state_value

    def update_policy(self) -> None:
        for row in range(self.max_row + 1):
            for col in range(self.max_column + 1):
                if (row, col) in self.terminal_states:
                    continue
                state_action_values = {
                    action: self.get_action_value(row, col, action, 1)
                    for action in self.policy[(row, col)].keys()
                }
                greedy_action = max(state_action_values, key=state_action_values.get)
                greedy_actions = [
                    action
                    for action, action_value in state_action_values.items()
                    if action_value == state_action_values[greedy_action]
                ]
                for action in self.policy[(row, col)].keys():
                    self.policy[(row, col)][action] = (
                        1 / len(greedy_actions) if action in greedy_actions else 0
                    )
        self.random_policy = False

    def take_actions(self, inplace_update: typing.Optional[bool] = False) -> np.array:
        next_values = np.zeros((self.max_row + 1, self.max_column + 1))
        for row in range(self.max_row + 1):
            for col in range(self.max_column + 1):
                if (row, col) in self.terminal_states:
                    continue
                state_value = self.take_actions_for_a_state(row, col)
                next_values[(row, col)] = state_value
                if inplace_update:
                    self.values[(row, col)] = state_value
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
