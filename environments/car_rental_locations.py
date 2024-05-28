class CarRentalLocation:
    def __init__(self, requests_lambda: float, returns_lambda: float) -> None:
        self.number_of_cars = 20
        self.requests_lambda = requests_lambda
        self.returns_lambda = returns_lambda
    

    def get_number_of_cars(self, poisson_lambda: float) -> int:
        pass
    
    def request_cars(self) -> None:
        pass

    def return_cars(self) -> None:
        pass
