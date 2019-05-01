import threading
import logging
import math

import kabc


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s')

class KArmedBanditProblem(threading.Thread):
    """
    Basic problem class for the k-Armed Bandit problem.
    It provides the basic interface.
    The solve method should be implemented by
    specific solvers that inherite from this class.

    """

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
        self.optimal_action_percent = kwargs["optimal_action_percent"]

        if "optimistic_Q_a" in kwargs:
            self.optimistic_Q_a = kwargs["optimistic_Q_a"]
        else:
            self.optimistic_Q_a = 0.0

        self.bandits = [kabc.KArmedBandit() for i in range(self.k)]
        self.optimal_action_index = self.get_optimal_action_index()
        self.optimal_action_counter = 0

        # Q_a - initial estimate for action a.

        self.Q_a = [self.optimistic_Q_a for i in range(self.k)]
        self.N_a = [0.0 for i in range(self.k)]

    def get_optimal_action_index(self):

        optimal_action_index = None
        optimal_action = -math.inf
        for i in range(self.k):
            if self.bandits[i].expected_reward > optimal_action:
                optimal_action = self.bandits[i].expected_reward
                optimal_action_index = i
        return optimal_action_index

    def print_bandit_optimal_rewards(self):
        s = ""
        for i in range(self.k):
            s = s + "bandit: {0}, action_value: {1}, ".format(i, self.bandits[i].expected_reward)
        logging.debug(s)

    def solve(self):
        pass

    def run(self):
        logging.debug("{0}".format(threading.current_thread()))
        self.solve()
        logging.debug("Ending...")
        return
