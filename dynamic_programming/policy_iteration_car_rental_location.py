import copy
import json
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
                cars: 1 if cars == 0 else 0
                for cars in range(
                    cars_movable_to_location_2, -1 - cars_movable_to_location_1, -1
                )
                # if cars >= -5 and cars <= 5
            }
    save_original_policy(policy)
    return policy


def evaluate_policy(env: car_rental_locations.CarsRentalLocations) -> None:
    theta = 50
    evaluation_iterations = 0
    while True:
        theta /= 10
        evaluation_iterations += 1
        current_values = copy.deepcopy(env.values)
        env.evaluate_policy()
        diffs = env.values - current_values
        delta = np.abs(diffs).max()
        if delta < theta:
            break
    print(
        f"Policy evaluation with in-place update completed after {evaluation_iterations} steps"
    )


def improve_policy(env: car_rental_locations.CarsRentalLocations) -> bool:
    current_policy = copy.deepcopy(env.policy)
    env.update_policy()
    return current_policy != env.policy


def save_original_policy(policy: dict) -> None:
    policy_to_save = {}
    for locations, actions_probs in policy.items():
        policy_to_save[f"({locations[0]}, {locations[1]})"] = actions_probs
    with open("dynamic_programming/car_rental__original_policy.json", "w") as fp:
        json.dump(policy_to_save, fp, indent=4)


def save_policy(policy: dict) -> None:
    policy_to_save = {}
    for locations, actions_probs in policy.items():
        policy_to_save[f"({locations[0]}, {locations[1]})"] = list(
            actions_probs.keys()
        )[list(actions_probs.values()).index(1)]
    with open("dynamic_programming/car_rental_policy.json", "w") as fp:
        json.dump(policy_to_save, fp, indent=4)


def iterate_policy() -> None:
    policy = get_initial_policy()
    env = car_rental_locations.CarsRentalLocations(policy)
    num_of_policy_iterations = 0
    while True:
        num_of_policy_iterations += 1
        print(f"Policy iteration {num_of_policy_iterations}")
        evaluate_policy(env)
        if not improve_policy(env):
            break
    save_policy(env.policy)
    print(env.values)
    print(
        f"Policy iteration completed after {num_of_policy_iterations} policy iterations"
    )


if __name__ == "__main__":
    iterate_policy()
