import math
import copy
from logger import *


BOARD_SIZE = 4

class State:
    def __init__(self, i, j, n=BOARD_SIZE) -> None:

        self.i = min(max(0, i), BOARD_SIZE - 1)
        self.j = min(max(0, j), BOARD_SIZE - 1) 
        self.n = n
        self.id = self.i * n + self.j

    def __str__(self) -> str:
        return f"{self.i} {self.j} {self.id}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        if self.i == o.i and self.j == o.j:
            return True
        else:
            return False

class Action:
    def __init__(self, di, dj) -> None:
        self.di = di
        self.dj = dj

    def __str__(self) -> str:
        return f"{self.di} {self.dj}"

    def __repr__(self) -> str:
        return self.__str__()


DEFAULT_TERMINAL_STATES = [State(0, 0), State(BOARD_SIZE-1, BOARD_SIZE-1)]


def make_states(terminal_states=DEFAULT_TERMINAL_STATES):

    states = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):

            s = State(i ,j)

            if s in terminal_states:
                continue

            states.append(s)

    return states

def print_matrix(v):

    for i in range(len(v)):
        for j in range(len(v[i])):
            s = str( round(v[i][j], 1) ).ljust(4, " ")
            print(f"{s} ", end=" ")
        print()

def approximate_state_value_function_for_random_policy(s, approx_v, terminal_states=DEFAULT_TERMINAL_STATES):

    actions = [Action( 0, -1),
               Action( 0,  1),
               Action(-1,  0),
               Action( 1,  0)]

    v_s = 0.0
    for a in actions:

        ni = s.i + a.di
        nj = s.j + a.dj

        r = -1
        gamma = 1.0
        s_prime = State(ni, nj)

        # if s_prime in terminal_states:
        #     r = 0.0

        # printd(f"a: {a} s: {s} -> s_prime: {s_prime} r: {r}")

        v_s += (1.0/4.0)*(r + gamma*approx_v[s_prime.i][s_prime.j])

    # printd(f"v_s: {v_s}")
    return v_s


def evaluate_policy(approx_v, states, theta=0.001):

    k = 0
    while True:

        print(f"k = {k}")
        print_matrix(approx_v)
        k += 1

        delta = 0.0

        new_approx_v = copy.deepcopy(approx_v)
        for s in states:

            v = approx_v[s.i][s.j]
            new_approx_v[s.i][s.j] = approximate_state_value_function_for_random_policy(s, approx_v)

            # printd(f"Processing state: {s} v: {v} approx_v[s.i][s.j]: {approx_v[s.i][s.j]} --> {v - approx_v[s.i][s.j]}")

            delta = max(delta, abs(v - new_approx_v[s.i][s.j]))

        approx_v = copy.deepcopy(new_approx_v)
        # printd(f"delta: {delta}")

        if delta < theta:
            break

        input()

    return approx_v

approx_v = [[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


states = make_states()
printd(states)

evaluate_policy(approx_v, states, theta=0.0001)