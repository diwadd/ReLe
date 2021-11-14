import copy
import numpy as np
import math

CAR_RENT_PRICE = 10
CAR_MOVE_PRICE = 2

RENTAL_REQ_FIRST_LOC = 3.0
MAX_POISSON_REQS_FIRST_LOC = 12

RENTAL_REQ_SECOND_LOC = 4.0
MAX_POISSON_REQS_SECOND_LOC = 14

RETURN_REQ_FIRST_LOC = 3.0
MAX_POISSON_RETURNS_FIRST_LOC = 12

RETURN_REQ_SECOND_LOC = 2.0
MAX_POISSON_RETURNS_SECOND_LOC = 10

MAX_CARS_AT_ANY_LOC = 20
MAX_CARS_MOVED_IN_ONE_NIGHT = 5

GAMMA = 0.9

def poisson(l, k):
    return math.exp(-l) * math.pow(l, k) / math.factorial(k)

def generate_poisson_array(l, s):

    poisson_array = [0 for _ in range(s+1)]
    for i in range(s+1):
        poisson_array[i] = poisson(l, i)

    return poisson_array

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
        for j in range(0, MAX_CARS_AT_ANY_LOC+1):
            states.append(State(i, j))

    return states

RENTAL_REQ_FIRST_LOC_POISSON = generate_poisson_array(RENTAL_REQ_FIRST_LOC, MAX_POISSON_REQS_FIRST_LOC)
RENTAL_REQ_SECOND_LOC_POISSON = generate_poisson_array(RENTAL_REQ_SECOND_LOC, MAX_POISSON_REQS_SECOND_LOC)

RETURN_REQ_FIRST_LOC_POISSON = generate_poisson_array(RETURN_REQ_FIRST_LOC, MAX_POISSON_RETURNS_FIRST_LOC)
RETURN_REQ_SECOND_LOC_POISSON = generate_poisson_array(RETURN_REQ_SECOND_LOC, MAX_POISSON_RETURNS_SECOND_LOC)

def approximate_state_value_function(s, approx_v, action, gamma=0.9):

    r = 0

    r += CAR_MOVE_PRICE * abs(action)

    # Cars at location one/two at the morning
    cars_loc_one_morning = min(max(s.n_cars_at_loc_one - action, 0), MAX_CARS_AT_ANY_LOC)
    cars_loc_two_morning = min(max(s.n_cars_at_loc_two - action, 0), MAX_CARS_AT_ANY_LOC)

    # We calculate the expected reward for the amount of cars for the given day.
    for rented_one in range(MAX_POISSON_REQS_FIRST_LOC+1): # Cars rented at location one.
        for rented_two in range(MAX_POISSON_REQS_SECOND_LOC+1): # Cars rented at location two.

            # We cannot rent more cars then we have.
            actual_rented_cars_one = int(min(cars_loc_one_morning, rented_one))
            actual_rented_cars_two = int(min(cars_loc_two_morning, rented_two))

            # We rented actual_rented_cars_one + actual_rented_cars_two so we earned:
            earnings = (actual_rented_cars_one + actual_rented_cars_two) * CAR_RENT_PRICE

            # Number of cars returned during the day (returned cars are avaialbe on the next day).
            for returned_one in range(MAX_POISSON_RETURNS_FIRST_LOC+1):
                for returned_two in range(MAX_POISSON_RETURNS_SECOND_LOC+1):

                    cars_at_eob_one = int(min(cars_loc_one_morning - actual_rented_cars_one + returned_one, MAX_CARS_AT_ANY_LOC))
                    cars_at_eob_two = int(min(cars_loc_one_morning - actual_rented_cars_one + returned_one, MAX_CARS_AT_ANY_LOC))

                    # p = poisson(RENTAL_REQ_FIRST_LOC, rented_one) * \
                    #     poisson(RENTAL_REQ_SECOND_LOC, rented_two) * \
                    #     poisson(RETURN_REQ_FIRST_LOC, returned_one) * \
                    #     poisson(RETURN_REQ_SECOND_LOC, returned_two)

                    p = RENTAL_REQ_FIRST_LOC_POISSON[rented_one] * \
                        RENTAL_REQ_SECOND_LOC_POISSON[rented_two] * \
                        RETURN_REQ_FIRST_LOC_POISSON[returned_one] * \
                        RETURN_REQ_SECOND_LOC_POISSON[returned_two]

                    # print(f"{rented_one} {rented_two} {returned_one} {returned_two} {actual_rented_cars_one} {actual_rented_cars_two} {cars_at_eob_one} {cars_at_eob_two}")

                    r += p * (earnings + gamma * approx_v[cars_at_eob_one][cars_at_eob_two])

    return r



def evaluate_policy(approx_v, states, policy, theta=0.001, gamma=0.9):

    k = 0
    while True:

        delta = 0.0

        for s in states:
            # print(f"State: {s}")
            i = s.n_cars_at_loc_one
            j = s.n_cars_at_loc_two
            action = policy[i][j]

            v = approx_v[i, j]
            approx_v[i, j] = approximate_state_value_function(s, approx_v, action)
            delta = max(delta, abs(v - approx_v[i, j]))

        print(f"Delta: {delta}")

        if delta < theta:
            break


actions = generate_actions()
print(actions)

states = createStateSpace()
# print(states)


policy = np.zeros((MAX_CARS_AT_ANY_LOC+1, MAX_CARS_AT_ANY_LOC+1))
approx_v = np.zeros((MAX_CARS_AT_ANY_LOC+1, MAX_CARS_AT_ANY_LOC+1))

print(policy.shape)
print(approx_v.shape)


evaluate_policy(approx_v, states, policy, theta=0.001, gamma=0.9)