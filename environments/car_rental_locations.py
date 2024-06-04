import typing

from scipy.stats import poisson
import numpy as np


np.random.seed(0)


class CarsRentalLocations:
    def __init__(
        self, policy: dict, discounting_factor: typing.Optional[float] = 0.9
    ) -> None:
        self.location_1_properties = {
            "renting_distribution": self.initialize_distribution(3),
            "return_distribution": self.initialize_distribution(3),
        }
        self.location_2_properties = {
            "renting_distribution": self.initialize_distribution(4),
            "return_distribution": self.initialize_distribution(2),
        }
        self.policy = policy
        self.values = np.zeros((21, 21))
        self.discounting_factor = discounting_factor

    def initialize_distribution(self, poisson_lambda: int) -> dict:
        return {i: poisson.pmf(i, poisson_lambda) for i in range(21)}

    def get_cars_at_the_end_of_the_day(
        self, remaining_cars: int, return_distribution: dict
    ) -> dict:
        cars_at_the_end_of_the_day = {}
        for i in range(0, 21 - remaining_cars):
            cars_at_the_end_of_the_day[remaining_cars + i] = return_distribution[
                remaining_cars + i
            ]
        return cars_at_the_end_of_the_day

    def get_location_states(
        self, available_cars: int, renting_distribution: dict, return_distribution: dict
    ) -> list:
        location_states = []
        for i in range(0, available_cars + 1):
            remaining_cars = available_cars - i
            cars_at_the_end_of_the_day = self.get_cars_at_the_end_of_the_day(
                remaining_cars, return_distribution
            )
            for end_day_cars, prob in cars_at_the_end_of_the_day.items():
                location_states.append(
                    {
                        "rented_cars": i,
                        "renting_distribution": renting_distribution[i],
                        "end_day_cars": end_day_cars,
                        "return_distribution": prob,
                    }
                )
        return location_states

    def get_state_action_value(
        self, state: tuple, moved_cars: int, probability: float
    ) -> float:
        if probability == 0:
            return 0
        next_state = (state[0] - moved_cars, state[1] + moved_cars)
        location_1_states = []
        location_1_states = self.get_location_states(
            next_state[0],
            self.location_1_properties["renting_distribution"],
            self.location_1_properties["return_distribution"],
        )
        location_2_states = self.get_location_states(
            next_state[1],
            self.location_2_properties["renting_distribution"],
            self.location_2_properties["return_distribution"],
        )
        cost_of_moving_cars = abs(moved_cars) * -2
        rental_income_and_next_state_value = 0
        for location_1_state in location_1_states:
            for location_2_state in location_2_states:
                rental_income = (
                    location_1_state["rented_cars"] + location_2_state["rented_cars"]
                ) * 10
                rental_income_and_next_state_value += (
                    location_1_state["renting_distribution"]
                    * location_2_state["renting_distribution"]
                    * location_1_state["return_distribution"]
                    * location_2_state["return_distribution"]
                    * (
                        rental_income
                        + self.discounting_factor
                        * self.values[
                            (
                                location_1_state["end_day_cars"],
                                location_2_state["end_day_cars"],
                            )
                        ]
                    )
                )
        action_value = probability * (
            cost_of_moving_cars + rental_income_and_next_state_value
        )
        return action_value

    def evaluate_policy(self) -> None:
        for location_1_cars in range(21):
            for location_2_cars in range(21):
                state = (location_1_cars, location_2_cars)
                state_policy = self.policy[state]
                state_value = 0
                for moved_cars, prob in state_policy.items():
                    action_value = self.get_state_action_value(state, moved_cars, prob)
                    state_value += action_value
                self.values[state] = state_value

    def update_policy(self) -> None:
        for state in self.policy.keys():
            state_action_values = {
                moved_cars: self.get_state_action_value(state, moved_cars, 1)
                for moved_cars in self.policy[state].keys()
            }
            greedy_action = max(state_action_values, key=state_action_values.get)
            # print("The greedy action for a state>>>", state, state_action_values, greedy_action)
            greedy_actions = [
                action
                for action, action_value in state_action_values.items()
                if action_value == state_action_values[greedy_action]
            ]
            for action in self.policy[state].keys():
                self.policy[state][action] = (
                    1 / len(greedy_actions) if action in greedy_actions else 0
                )
