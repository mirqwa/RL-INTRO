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

    def request_cars(self) -> None:
        requested_cars = min(
            self.number_of_cars, self.get_number_of_cars(self.rental_probability_dist)
        )
        self.rented_cars += requested_cars
        self.number_of_cars -= requested_cars

    def return_cars(self) -> None:
        returned_cars = min(
            self.rented_cars, self.get_number_of_cars(self.returns_probability_dist)
        )
        self.rented_cars -= returned_cars
        self.number_of_cars += returned_cars


def initialize_car_locations() -> None:
    location = CarRentalLocation(3, 3)
    location.request_cars()
    location.return_cars()


if __name__ == "__main__":
    initialize_car_locations()
