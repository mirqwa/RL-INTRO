import os
import sys

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


if __name__ == "__main__":
    policy = get_initial_policy()
    car_locations = car_rental_locations.CarRentalLocations(policy)
    car_locations.rent_cars()
    car_locations.evaluate_policy()
