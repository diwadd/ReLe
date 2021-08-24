import math

BOARD_SIZE = 4

state_value_function = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


class State:
    def __init__(self, i, j, n=BOARD_SIZE) -> None:
        self.i = i
        self.j = j
        self.n = n
        self.id = i * n + j

    def __str__(self) -> str:
        return f"{self.i} {self.j} {self.id}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        if self.i == o.i and self.j == o.j:
            return True
        else:
            return False

def make_states(terminal_states=[State(0, 0), State(BOARD_SIZE-1, BOARD_SIZE-1)]):

    states = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):

            s = State(i ,j)

            if s in terminal_states:
                continue

            states.append(s)

    return states


states = make_states()

def approximate_state_value_function_for_policy_pi(policy):
    pass

def evaluate_policy(approx_v, states, theta=0.001):

    while True:
        delta = 0.0

        for s in states:
            v = approx_v[s.i][s.j]
            approx_v[s.i][s.j] = approximate_state_value_function_for_policy_pi(policy)
            delta = max(delta, abs(v - approx_v[s.i][s.j]))

        if delta < theta:
            break

    return approx_v



for s in states:
    print(s)