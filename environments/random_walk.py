import numpy as np


np.random.seed(0)


class RandomWalk:
    def __init__(self, num_of_states: int, initial_state: int, max_steps: int) -> None:
        self.num_of_states = num_of_states
        self.terminal_states_rewards = {
            num_of_states: 1,
            0: -1,
        }
        self.initial_state = initial_state
        self.current_state = initial_state
        self.possible_steps = [i for i in range(-max_steps, max_steps + 1) if i != 0]
        self.steps_probabilities = [
            1 / (len(self.possible_steps)) for _ in self.possible_steps
        ]
        self.terminated = False
        self.reward = 0

    def take_a_step(self) -> None:
        steps = np.random.choice(self.possible_steps, 1, p=self.steps_probabilities)[0]
        self.current_state = (
            min(self.current_state + steps, self.num_of_states)
            if steps > 0
            else max(self.current_state + steps, 0)
        )
        if self.current_state in self.terminal_states_rewards.keys():
            self.reward = self.terminal_states_rewards[self.current_state]
            self.terminated = True
    
    def generate_episode(self):
        self.current_state = self.initial_state
        self.terminated = False
        self.reward = 0
        action_counts = 0
        while self.terminated is False:
            action_counts += 1
            self.take_a_step()
