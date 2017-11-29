import threading
import logging
import time
import random
import operator

import k_armed_bandit_class as kabc


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s')

class KArmedBanditProblem(threading.Thread):


    def __init__(self,
                 group=None,
                 target=None,
                 name=None,
                 args=(),
                 kwargs={},
                 daemon=None):

        threading.Thread.__init__(self,
                                  group=group,
                                  target=target,
                                  name=name,
                                  daemon=daemon)

        self.k = kwargs["k"]
        self.max_time = kwargs["max_time"]
        self.avg_reward = kwargs["avg_reward"]

        if "epsilon" in kwargs:
            self.epsilon = kwargs["epsilon"]
        else:
            self.epsilon = 0.0

        self.bandits = [kabc.KArmedBandit() for i in range(self.k)]
        self.Q_a = [0.0 for i in range(self.k)]
        self.N_a = [0.0 for i in range(self.k)]

    def print_bandit_optimal_rewards(self):
        s = ""
        for i in range(self.k):
            s = s + "bandit: {0}, action_value: {1}, ".format(i, self.bandits[i].action_value)
        logging.debug(s)


    def solve(self):
        logging.debug("Solving for k = {0}".format(self.k))

        # self.print_bandit_optimal_rewards()

        for t in range(self.max_time):

            # Get max action at time t. a_t is the index of the action.
            # With probability epsilon select random action.
            if random.random() > self.epsilon:
                a_t, _ = max(enumerate(self.Q_a), key=operator.itemgetter(1))
            else:
                a_t = random.randint(0, self.k - 1)

            # Perform action a_t. Pull the arm of the
            # a_t-th bandit.
            r_t = self.bandits[a_t].reward()

            self.N_a[a_t] = self.N_a[a_t] + 1.0
            # Update estimate for action a_t
            delta = (1.0/self.N_a[a_t])*(r_t - self.Q_a[a_t])
            self.Q_a[a_t] = self.Q_a[a_t] + delta

            self.avg_reward[t] = self.avg_reward[t] + r_t

    def run(self):
        logging.debug("{0}".format(threading.current_thread()))
        self.solve()
        logging.debug("Ending...")
        return