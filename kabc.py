import random
import threading
import operator
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from logger import *


class KArmedBanditLever:

    def __init__(self,
                 mu=0.0,
                 sigma=1.0):

        self.expected_reward = random.gauss(mu, sigma)
        # printd("self.expected_reward: {0}".format(self.expected_reward))

    def reward(self, sigma=1.0):
        return random.gauss(self.expected_reward, sigma)

    def __repr__(self):
        r = "%r, action_value: %s".format(self.__class__, self.expected_reward)
        return r

    def __str__(self):
        s = "KArmedBandit, action_value: {0}".format(self.expected_reward)
        return s


class KABProblemSolutionMethod(ABC):

    @abstractmethod
    def set_initial_values(self):
        pass

    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def update_number_of_times_action_was_chosen(self):
        pass

    @abstractmethod
    def update_estimated_value_of_action(self):
        pass


class GreedySolutionMethod(KABProblemSolutionMethod):
    """

    """
    def __init__(self, step_size=None):
        self.step_size = step_size

    def set_initial_values(self, k):
        return [0.0 for i in range(k)]

    def select_action(self, Q_t_a):
        # printd("Selecting action from: {0}".format(Q_t_a))
        qta = np.array(Q_t_a)
        qta = list(np.where(qta == np.amax(qta))[0])
        return random.choice(qta)
        #return Q_t_a.index(max(Q_t_a))

    def update_number_of_times_action_was_chosen(self, N_a=[], a=0):
        N_a[a] = N_a[a] + 1

    def update_estimated_value_of_action(self, Q_t_a=[], N_a=[], a=0, R_t=0):
        if self.step_size is None:
            Q_t_a[a] = Q_t_a[a] + (1.0/N_a[a])*(R_t - Q_t_a[a])
        else:
            Q_t_a[a] = Q_t_a[a] + self.step_size*(R_t - Q_t_a[a])


class EpsilonSolutionMethod(GreedySolutionMethod):
    """

    """
    def __init__(self, epsilon=0.1, step_size=None):
        self.epsilon = epsilon
        GreedySolutionMethod.__init__(self, step_size=step_size)

    def select_action(self, Q_t_a):
        p = random.uniform(0.0, 1.0)
        if p > self.epsilon:
            qta = np.array(Q_t_a)
            qta = list(np.where(qta == np.amax(qta))[0])
            return random.choice(qta)
            # return Q_t_a.index(max(Q_t_a))
        else:
            k = len(Q_t_a)
            return random.randint(0, k - 1)


class OIVSolutionMethod(EpsilonSolutionMethod):
    """
    Optimistic Initial Values (OIV) - solution method
    """
    def __init__(self, iv=5.0, epsilon=0.0, step_size=None):
        self.iv = iv
        EpsilonSolutionMethod.__init__(self, epsilon=epsilon, step_size=step_size)

    def set_initial_values(self, k):
        return [self.iv for i in range(k)]


class UCBSolutionMethod(OIVSolutionMethod):
    """
    Upper Confidence Bound (UCB) - solution method
    """
    def __init__(self, c=1.0, iv=5.0, epsilon=0.0, step_size=None):
        self.c = c
        OIVSolutionMethod.__init__(self, iv=iv, epsilon=epsilon, step_size=step_size)

    def select_action(self, Q_t_a):
        pass

class GBSolutionMethod:
    """
    Gradient Bandit - solution method
    """
    pass


class KArmedBanditProblem(threading.Thread):

    def __init__(self,
                 solution_method=None,
                 k=10,
                 t_start=1,
                 t_stop=100,
                 group=None,
                 target=None,
                 name=None,
                 daemon=None):

        assert solution_method is not None, "solution_method cannot be None"

        threading.Thread.__init__(self,
                                  group=group,
                                  target=target,
                                  name=name,
                                  daemon=daemon)

        self.solution_method = solution_method
        self.k = k
        self.t_start = t_start
        self.t_stop = t_stop

        # The k-armed bandit with k levers (k options or action to choose from).
        self.kab = [KArmedBanditLever() for a in range(self.k)]

        # Initial estimates of the action values q_star_a.
        self.Q_t_a = solution_method.set_initial_values(k=self.k)

        # Number of times a given action was chosen.
        # In other words number of times the k-armed bandit at a was chosen and played.
        self.N_a = [0 for a in range(self.k)]


        # Statistics

        self.optimal_action = self.kab.index(max(self.kab, key=operator.attrgetter("expected_reward")))
        # printd("optimal_action: {0}".format(self.optimal_action))

        # n_optimal_action  - number of times the optimal action was chosen.
        self.avg_reward = 0.0
        self.reward_at_t = [0.0 for i in range(self.t_start, self.t_stop + 1)]
        self.percent_optimal_action = [0.0 for i in range(self.t_start, self.t_stop + 1)]

        self._return = None

    def solve(self):
        for t in range(self.t_start, self.t_stop + 1):
            a = self.solution_method.select_action(self.Q_t_a)

            R_t = self.kab[a].reward()
            # printd("a: {0} R_t: {1}".format(a, R_t))

            self.solution_method.update_number_of_times_action_was_chosen(N_a=self.N_a, a=a)
            self.solution_method.update_estimated_value_of_action(Q_t_a=self.Q_t_a,
                                                                  N_a=self.N_a,
                                                                  a=a,
                                                                  R_t=R_t)

            # Update statistics
            self.reward_at_t[t - 1] = R_t
            # printd("average reward at t: {0}". format(self.avg_reward_at_t[t-1]))

            if a == self.optimal_action:
                self.percent_optimal_action[t-1] = 100

        return self.reward_at_t, self.percent_optimal_action

    def run(self):
        printd("{0}".format(threading.current_thread()))
        self._return = self.solve()
        printd("Ending...")
        return

    def join(self, *args, **kwargs):
        threading.Thread.join(self, *args, **kwargs)
        return self._return


class KArmedBanditTestbed:

    def __init__(self,
                 n=10,
                 solution_method=GreedySolutionMethod,
                 k=10,
                 t_start=1,
                 t_stop=100):

        assert t_start == 1, "Time must start at 1, i.e., t_start must be equal to 1."
        assert t_stop > t_start, "t_stop must be greater then t_start"
        # In such a case t_stop is equal to the number of time steps.

        self.t_start = t_start
        self.t_stop = t_stop

        self.n = n
        self.n_problems = [None for i in range(self.n)]
        self.results = [None for i in range(self.n)]
        self.avg_reward_at_t_testbed = np.zeros(t_stop)
        self.percent_optimal_action_testbed = np.zeros(t_stop)

        for i in range(n):
            kabp = KArmedBanditProblem(solution_method=solution_method, k=k, t_start=self.t_start, t_stop=self.t_stop)
            self.n_problems[i] = kabp

    def run(self):

        for p in range(self.n):
            self.n_problems[p].start()

        for p in range(self.n):
            self.results[p] = self.n_problems[p].join()

        for p in range(self.n):
            reward_at_t_p, percent_optimal_action_p = self.results[p]
            self.avg_reward_at_t_testbed = self.avg_reward_at_t_testbed + np.array(reward_at_t_p)
            self.percent_optimal_action_testbed = self.percent_optimal_action_testbed + np.array(percent_optimal_action_p)

        self.avg_reward_at_t_testbed = self.avg_reward_at_t_testbed/self.n
        self.percent_optimal_action_testbed = self.percent_optimal_action_testbed/self.n

        return self.avg_reward_at_t_testbed, self.percent_optimal_action_testbed


if __name__ == "__main__":

    n = 2000
    t_stop = 1000

    # Greedy
    greedy_sm = GreedySolutionMethod()
    greedy_kab_tb = KArmedBanditTestbed(solution_method=greedy_sm, n=n, t_stop=t_stop)
    greedy_avg_reward_at_t_testbed, greedy_percent_optimal_action_testbed = greedy_kab_tb.run()

    # Epsilon = 0.1
    e0p1_sm = EpsilonSolutionMethod(epsilon=0.1)
    e0p1_kab_tb = KArmedBanditTestbed(solution_method=e0p1_sm, n=n, t_stop=t_stop)
    e0p1_avg_reward_at_t_testbed, e0p1_percent_optimal_action_testbed = e0p1_kab_tb.run()

    # Epsilon = 0.01
    e0p01_sm = EpsilonSolutionMethod(epsilon=0.01)
    e0p01_kab_tb = KArmedBanditTestbed(solution_method=e0p01_sm, n=n, t_stop=t_stop)
    e0p01_avg_reward_at_t_testbed, e0p01_percent_optimal_action_testbed = e0p01_kab_tb.run()

    oiv_sm = OIVSolutionMethod(iv=5.0, epsilon=0.0, step_size=0.1)
    oiv_kab_tb = KArmedBanditTestbed(solution_method=oiv_sm, n=n, t_stop=t_stop)
    oiv_avg_reward_at_t_testbed, oiv_percent_optimal_action_testbed = oiv_kab_tb.run()

    e0p1_step0p1_sm = EpsilonSolutionMethod(epsilon=0.1, step_size=0.1)
    e0p1_step0p1_kab_tb = KArmedBanditTestbed(solution_method=e0p1_step0p1_sm, n=n, t_stop=t_stop)
    e0p1_step0p1_avg_reward_at_t_testbed, e0p1_step0p1_percent_optimal_action_testbed = e0p1_step0p1_kab_tb.run()

    plt.plot(greedy_avg_reward_at_t_testbed, 'g')
    plt.plot(e0p1_avg_reward_at_t_testbed, 'b')
    plt.plot(e0p01_avg_reward_at_t_testbed, 'r')
    plt.ylabel("Average reward [a. u.]")
    plt.ylim(0, 1.5)
    plt.show()

    plt.plot(greedy_percent_optimal_action_testbed, 'g')
    plt.plot(e0p1_percent_optimal_action_testbed, 'b')
    plt.plot(e0p01_percent_optimal_action_testbed, 'r')
    plt.ylabel("Optimal action [%]")
    plt.ylim(0, 110)
    plt.show()

    plt.plot(oiv_percent_optimal_action_testbed, 'b')
    plt.plot(e0p1_step0p1_percent_optimal_action_testbed, 'k')
    plt.ylabel("Optimal action [%]")
    plt.ylim(0, 110)
    plt.show()