import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import gridworld


if __name__ == "__main__":
    env = gridworld.GridWorld(4, 4, max_iterations=10000)
    env.iterate_states()
    print(env.values)