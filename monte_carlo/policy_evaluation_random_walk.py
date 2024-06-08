import json
import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import random_walk


def save_state_values(state_values: dict) -> None:
    with open("monte_carlo/random_walk_state_values.json", "w") as fp:
        json.dump(state_values, fp, indent=4)


def evaluate_policy() -> None:
    env = random_walk.RandomWalk(1000, 500, 100)
    state_values = {state: {"count": 0, "average": 0} for state in range(1, 1001)}
    discounting_factor = 1
    for _ in range(2000000):
        states, rewards = env.generate_episode()
        G = 0
        for i in range(len(states) - 2, -1, -1):
            state = states[i]
            next_reward = rewards[i]
            G = discounting_factor * G + next_reward
            if state not in states[:i]:
                state_values[state]["count"] += 1
                state_values[state]["average"] += (
                    G - state_values[state]["average"]
                ) / state_values[state]["count"]
    # state_values = {state: value["average"] for state, value in state_values.items()}
    save_state_values(state_values)


if __name__ == "__main__":
    evaluate_policy()
