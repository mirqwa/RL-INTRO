import numpy as np


class GridWorld:
    def __init__(self, rows, columns, max_iterations=3) -> None:
        self.max_row = rows - 1
        self.max_column = columns - 1
        self.values = np.zeros((rows, columns))
        self.terminal_states = [(0, 0), (self.max_row, self.max_column)]
        self.iterations = 0
        self.max_iterations = max_iterations
        self.valid_actions = ["up", "right", "down", "left"]

    def take_actions_for_a_state(self, row, column) -> None:
        for action in self.valid_actions:
            if action == "up":
                new_state = (max(row - 1, 0), column)
            elif action == "right":
                new_state = (row, min(column + 1, self.max_column))
            elif action == "down":
                new_state = (min(row + 1, self.max_row), column)
            else:
                new_state = (row, max(column - 1, 0))
            print((row, column), action, new_state)

    def get_action_reward(self, next_state) -> None:
        pass

    def iterate_states(self) -> None:
        for i in range(self.max_iterations):
            for row in range(self.max_row + 1):
                for col in range(self.max_column + 1):
                    if (row, col) in self.terminal_states:
                        continue
                    self.take_actions_for_a_state(row, col)


if __name__ == "__main__":
    env = GridWorld(4, 4)
    env.iterate_states()
