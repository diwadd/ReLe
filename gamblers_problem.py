from email import policy
import math
import matplotlib.pyplot as plt

MIN_CAPITAL = 1
MAX_CAPITAL = 99

TERMINATION_STATE_LOST = 0
TERMINATION_STATE_WIN = 101

def update_state_value(state, state_value_function, gamma=1.0, p_h=0.4):

    actions = [i for i in range(1, state+1)]

    p = [p_h, 1 - p_h]

    maximum_value = 0
    maximum_action = 0

    for a in actions:
        v = 0.0

        sp_win = min(state + a, TERMINATION_STATE_WIN)
        sp_loss = max(state - a, TERMINATION_STATE_LOST)

        states_prime = [sp_win, sp_loss]
        for i in range(len(states_prime)):

            sp = states_prime[i]

            r = 0.0
            if sp == TERMINATION_STATE_WIN:
                r = 1.0

            v += p[i] * (r + gamma * state_value_function[sp])

        if v >= maximum_value:
            maximum_value = v
            maximum_action = a

    return maximum_value, maximum_action


def value_iteration(states, state_value_function, gamma=1.0, theta=0.001):

    index = 0
    while True:
        delta = 0.0

        print(f"Iteration: {index}")

        for i in range(1, len(states)):

            s = states[i]
            v = state_value_function[s]
            state_value_function[s], _ = update_state_value(s, state_value_function, gamma)
            delta = max(delta, abs(v - state_value_function[s]))

        if delta < theta:
            break
    
        index += 1


def get_deterministic_policy(states, state_value_function, deterministic_policy, gamma=1.0):

    for s in states:

        _, a = update_state_value(s, state_value_function, gamma)
        deterministic_policy[s] = a

states = [i for i in range(TERMINATION_STATE_LOST, TERMINATION_STATE_WIN+1)]
state_value_function = [0.0 for _ in range(TERMINATION_STATE_LOST, TERMINATION_STATE_WIN+1)]
deterministic_policy = [0.0 for _ in range(TERMINATION_STATE_LOST, TERMINATION_STATE_WIN+1)]

print(states)

state_value_function[TERMINATION_STATE_LOST] = 0.0
state_value_function[TERMINATION_STATE_WIN] = 1.0

value_iteration(states, state_value_function)

print(state_value_function)

m = max(state_value_function[1:-1])
for i in range(0, len(state_value_function)):
    state_value_function[i] /= m


get_deterministic_policy(states, state_value_function, deterministic_policy, gamma=1.0)


plt.plot(state_value_function[1:-1])
plt.show()

plt.plot(deterministic_policy[1:-1])
plt.show()