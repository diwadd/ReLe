import random
import operator
import logging
from k_armed_bandit_problem import KArmedBanditProblem

class KArmedBanditEpsilonGreedy(KArmedBanditProblem):

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

        if "epsilon" in kwargs:
            self.epsilon = kwargs["epsilon"]
        else:
            self.epsilon = 0.0

    def solve(self):
        logging.debug("Starting solver...")

        # Time is counted from 1 to self.max_time.
        # Range work ony to self.max_time if no + 1.
        for t in range(1, self.max_time + 1):

            # Get max action at time t. a_t is the index of the action.
            # With probability epsilon select random action.
            if random.random() > self.epsilon:
                a_t, _ = max(enumerate(self.Q_a), key=operator.itemgetter(1))
            else:
                a_t = random.randint(0, self.k - 1)

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