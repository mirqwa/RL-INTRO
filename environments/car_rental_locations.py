import typing

from scipy.stats import poisson
import numpy as np


class CarRentalLocation:
    def __init__(self, rental_lambda: float, returns_lambda: float) -> None:
        self.number_of_cars = 20
        self.rental_probability_dist = self.initialize_distribution(rental_lambda)
        self.returns_probability_dist = self.initialize_distribution(returns_lambda)
        self.rented_cars = 0

    def initialize_distribution(self, poisson_lambda: int) -> dict:
        return {i: poisson.pmf(i, poisson_lambda) for i in range(20)}

    def get_number_of_cars(self, rental_probability_dist: dict) -> int:
        rental_requests = np.random.choice(
            list(rental_probability_dist.keys()),
            1,
            p=list(rental_probability_dist.values()),
        )
        return rental_requests[0]

    def rent_cars(self) -> typing.Tuple[int]:
        requested_cars = min(
            self.number_of_cars, self.get_number_of_cars(self.rental_probability_dist)
        )
        self.rented_cars += requested_cars
        prev_number_of_cars = self.number_of_cars
        self.number_of_cars -= requested_cars
        return prev_number_of_cars - 1, requested_cars * 10

    def return_cars(self) -> None:
        returned_cars = min(
            self.rented_cars, self.get_number_of_cars(self.returns_probability_dist)
        )
        self.rented_cars -= returned_cars
        self.number_of_cars += returned_cars


class CarRentalLocations:
    def __init__(
        self, policy: dict, discounting_factor: typing.Optional[float] = 0.9
    ) -> None:
        self.location_1 = CarRentalLocation(3, 3)
        self.location_2 = CarRentalLocation(4, 2)
        self.policy = policy
        self.values = np.zeros((21, 21))
        self.discounting_factor = discounting_factor

    def rent_cars(self) -> None:
        location_1_state, location_1_state_value = self.location_1.rent_cars()
        location_2_state, location_2_state_value = self.location_2.rent_cars()
        self.values[(location_1_state, location_2_state)] = (
            location_1_state_value + location_2_state_value
        )

    def get_state_action_value(
        self, state: tuple, moved_cars: int, probability: float
    ) -> float:
        next_state = (state[0] - moved_cars, state[1] + moved_cars)
        return probability * (
            abs(moved_cars) * -2 + self.discounting_factor * self.values[next_state]
        )

    def evaluate_policy(self) -> None:
        for state, probs in self.policy.items():
            cost_of_moving_cars = 0
            for moved_cars, prob in probs.items():
                cost_of_moving_cars += self.get_state_action_value(
                    state, moved_cars, prob
                )
            self.values[state] += cost_of_moving_cars

    def update_policy(self) -> None:
        for state in self.policy.keys():
            state_action_values = {
                moved_cars: self.get_state_action_value(state, moved_cars, 1)
                for moved_cars in self.policy[state].keys()
            }
            greedy_action = max(state_action_values, key=state_action_values.get)
            greedy_actions = [
                action
                for action, action_value in state_action_values.items()
                if action_value == state_action_values[greedy_action]
            ]
            for action in self.policy[state].keys():
                self.policy[state][action] = (
                    1 / len(greedy_actions) if action in greedy_actions else 0
                )
