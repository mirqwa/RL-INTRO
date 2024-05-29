import copy
import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))

from environments import car_rental_locations


def get_initial_policy():
    policy = {}
    for location_1_cars in range(21):
        for location_2_cars in range(21):
            cars_movable_to_location_2 = min(location_1_cars, 20 - location_2_cars)
            cars_movable_to_location_1 = min(location_2_cars, 20 - location_1_cars)
            policy[(location_1_cars, location_2_cars)] = {
                cars: 0
                for cars in range(
                    cars_movable_to_location_2, -1 - cars_movable_to_location_1, -1
                )
            }
    return policy


def evaluate_policy(env: car_rental_locations.CarRentalLocations) -> None:
    num_of_iterations = 1
    theta = 0.1
    while True:
        current_avg_values = copy.deepcopy(env.values) / num_of_iterations
        num_of_iterations += 1
        env.rent_cars()
        env.move_cars()
        diffs = env.values / num_of_iterations - current_avg_values
        delta = np.abs(diffs).max()
        if delta < theta:
            break
    print(
        f"Policy evaluation with in-place update completed after {num_of_iterations} steps"
    )


def iterate_policy() -> None:
    policy = get_initial_policy()
    env = car_rental_locations.CarRentalLocations(policy)
    while True:
        evaluate_policy(env)
        break
    



if __name__ == "__main__":
    iterate_policy()
