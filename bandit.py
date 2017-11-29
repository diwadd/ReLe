import random
import operator
import math

import numpy as np

class Bandit:
    def __init__(self):
        self.q_star_a = random.gauss(0.0, 1.0)

    def get_reward(self):
        r_t = random.gauss(self.q_star_a, 1.0)
        return r_t

    def __str__(self):
        return str(self.q_star_a)

class KMultiBandit:

    def __init__(self, k):
        self.k = k
        self.bandits = [Bandit() for i in range(k)]
        self.a_reward_sum = [0.0 for i in range(k)]
        self.n_a_calls = [0.0 for i in range(k)]
        self.q_t_a_estimates = [0.0 for i in range(k)]
        self.total_reward_vs_time = None
        self.total_reward = 0.0
        self.t_stop = None

        self.optimal_bandit = -1.0*math.inf
        for i in range(k):
            if self.bandits[i].q_star_a > self.optimal_bandit:
                self.optimal_bandit = self.bandits[i].q_star_a


    def run_simulation(self, t_stop=10, epsilon=0.0):

        self.t_stop = t_stop
        self.total_reward_vs_time = [0.0 for i in range(t_stop)]
        for t in range(t_stop):
            # Get the greedy action.
            if random.random() > epsilon:
                a_index, _ = max(enumerate(self.q_t_a_estimates), key=operator.itemgetter(1))
            else:
                a_index = random.randint(0, self.k - 1)

            # Get reward for given action.
            r_t = self.bandits[a_index].get_reward()

            self.total_reward = self.total_reward + r_t
            self.total_reward_vs_time[t] = r_t

            self.a_reward_sum[a_index] = self.a_reward_sum[a_index] + r_t
            self.n_a_calls[a_index] = self.n_a_calls[a_index] + 1.0

            delta = self.a_reward_sum[a_index]/self.n_a_calls[a_index]
            self.q_t_a_estimates[a_index] = self.q_t_a_estimates[a_index] + delta
        return self.total_reward_vs_time

    def run_multiple_simulations(self, t_stop=10, epsilon=0.0, n=10):

        self.reward_distribution()

        self.average_reward = np.zeros((t_stop))
        for i in range(n):
            self.total_reward_vs_time = self.run_simulation(t_stop=t_stop, epsilon=epsilon)
            self.average_reward = self.average_reward + np.array(self.total_reward_vs_time)

        self.average_reward = self.average_reward/n

        return self.average_reward

    def reward_distribution(self):
        for i in range(len(self.bandits)):
            print("Bandit mean reward: {0}".format(self.bandits[i]))