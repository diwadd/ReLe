import copy
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
    def __init__(self, number):
        self.number = number

    def __str__(self):
        return f"Cars moved: {self.number}"

    def __repr__(self):
        return self.__str__()


def generate_actions():

    actions = []
    for i in range(-MAX_CARS_MOVED_IN_ONE_NIGHT, MAX_CARS_MOVED_IN_ONE_NIGHT+1):
        actions.append(Action(number=i))

    return actions


class State:
    def __init__(self, n_cars_at_loc_one, n_cars_at_loc_two):
        self.n_cars_at_loc_one = n_cars_at_loc_one
        self.n_cars_at_loc_two = n_cars_at_loc_two

    def __str__(self):
        return f"({self.n_cars_at_loc_one},{self.n_cars_at_loc_two})"

    def __repr__(self):
        return self.__str__()


def createStateSpace():

    states = []
    for i in range(0, MAX_CARS_AT_ANY_LOC+1):
        states.append(State(i, MAX_CARS_AT_ANY_LOC - i))

    return states


def evaluate_policy(approx_v, states, theta=0.001, gamma=0.9):

    k = 0
    while True:

        delta = 0.0
        new_approx_v = copy.deepcopy(approx_v)

        for s in states:
            v = approx_v[s.n_cars_at_loc_one, s.n_cars_at_loc_two]



actions = generate_actions()
print(actions)

states = createStateSpace()
print(states)


pi = np.ones((MAX_CARS_AT_ANY_LOC, MAX_CARS_AT_ANY_LOC)) / (MAX_CARS_AT_ANY_LOC ** 2)
approx_v = [[0.0 for _ in range(MAX_CARS_AT_ANY_LOC)] for _ in range(MAX_CARS_AT_ANY_LOC)]

# print(pi)