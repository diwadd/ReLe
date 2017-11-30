from k_armed_bandit_epsilon_greedy import KArmedBanditEpsilonGreedy


class KArmedBanditTestbed:

    def __init__(self,
                 bandit_problem_class=KArmedBanditEpsilonGreedy,
                 k=10,
                 n_problems=2000,
                 max_time=1000,
                 kwargs={"epsilon": 0.0}):

        self.max_time = max_time
        self.n_problems = n_problems
        self.bandit_threads = [None for i in range(n_problems)]

        # A Python list is thread safe. We will use
        # it to track the average reward.
        self.avg_reward = [0.0 for i in range(max_time)]

        # bandit_problem_class specific kwargs.
        self.kwargs = kwargs

        # General kwargs for all bandit_problem_classes.
        self.kwargs["k"] = k
        self.kwargs["max_time"] = self.max_time
        self.kwargs["n_problems"] = self.n_problems
        self.kwargs["avg_reward"] = self.avg_reward

        # Construct the problems. Each problem
        # is in fact a single thread.
        for i in range(self.n_problems):
            self.bandit_threads[i] = bandit_problem_class(kwargs=kwargs)

    def run(self):
        # Start the threads.
        for i in range(self.n_problems):
            self.bandit_threads[i].start()

        # Wait until the all the threads finish.
        for i in range(self.n_problems):
            self.bandit_threads[i].join()

        # Calculate the average over all the problems.
        for t in range(self.max_time):
            self.avg_reward[t] = self.avg_reward[t]/self.n_problems

        return self.avg_reward
