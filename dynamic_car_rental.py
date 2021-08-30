import numpy as np

CAR_RENT_PRICE = 10
CAR_MOVE_PRICE = 2

RENTAL_REQ_FIRST_LOC = 3.0
RENTAL_REQ_SECOND_LOC = 4.0
RETURN_REQ_FIRST_LOC = 3.0
RETURN_REQ_SECOND_LOC = 2.0

MAX_CARS_AT_ANY_LOC = 20
MAX_CARS_MOVED_IN_ONE_NIGHT = 5

GAMMA = 0.9

class Action:
    def __init__(self, from_loc, to_loc, number):
        self.from_loc = from_loc
        self.to_loc = to_loc
        self.number = number

    def __str__(self):
        return f"{self.from_loc} -> {self.to_loc} - {self.number}"

    def __repr__(self):
        return self.__str__()


def generate_actions():

    actions = []
    for i in range(MAX_CARS_MOVED_IN_ONE_NIGHT+1):
        actions.append(Action(from_loc=1, to_loc=2, number=i))

    for i in range(MAX_CARS_MOVED_IN_ONE_NIGHT+1):
        actions.append(Action(from_loc=2, to_loc=1, number=i))

    return actions

actions = generate_actions()

print(actions)