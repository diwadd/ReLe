import random
import operator
import logging
import math
from k_armed_bandit_problem import KArmedBanditProblem

class KArmedBanditUCB(KArmedBanditProblem):
    # UCB - Upper Confidence Bound

    def __init__(self,
                 group=None,
                 target=None,
                 name=None,
                 args=(),
                 kwargs={},
                 daemon=None):

        KArmedBanditProblem.__init__(self,
                                     group=group,
                                     target=target,
                                     name=name,
                                     args=args,
                                     kwargs=kwargs,
                                     daemon=daemon)

        if "c" in kwargs:
            self.c = kwargs["c"]
        else:
            # With c = 0.0 UCB is equivalent to
            # the epsilon greedy solver.
            self.c = 0.0

    def solve(self):
        logging.debug("Starting solver...")

        # Time is counted from 1 to self.max_time.
        # Range work ony to self.max_time if no + 1.
        for t in range(1, self.max_time + 1):

            # The Upper Confidence Bound of choosing the most
            # optimal action.
            Q_ucb = [self.Q_a[i] for i in range(self.k)]
            for i in range(self.k):
                if self.N_a[i] != 0:
                    Q_ucb[i] = Q_ucb[i] + self.c * math.sqrt(math.log(t) / self.N_a[i])
                else:
                    # If N_a[i] = 0 then a given by index i is the
                    # maximizing action.
                    Q_ucb[i] = Q_ucb[i] + math.inf

            a_t, _ = max(enumerate(Q_ucb), key=operator.itemgetter(1))

            if a_t == self.optimal_action_index:
                self.optimal_action_counter = self.optimal_action_counter + 1
                self.optimal_action_percent[t] = self.optimal_action_percent[t] + float(self.optimal_action_counter/t)
                #self.optimal_action_percent[t] = self.optimal_action_percent[t] + 1.0

            # Perform action a_t. Pull the arm of the
            # a_t-th bandit.
            r_t = self.bandits[a_t].reward()

            self.N_a[a_t] = self.N_a[a_t] + 1.0
            # Update estimate for action a_t
            delta = (1.0/self.N_a[a_t])*(r_t - self.Q_a[a_t])
            self.Q_a[a_t] = self.Q_a[a_t] + delta

            self.avg_reward[t] = self.avg_reward[t] + r_t