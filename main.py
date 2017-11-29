import matplotlib.pyplot as plt

import k_armed_bandit_problem as kabp

k = 10
n_problems = 2000
max_time = 1000

# List are thread safe.
avg_reward = [0.0 for i in range(max_time)]
bandit_threads = [None for i in range(n_problems)]

for i in range(n_problems):
    p = kabp.KArmedBanditProblem(kwargs={"k": k,
                                         "epsilon": 0.0,
                                         "max_time": max_time,
                                         "avg_reward": avg_reward})
    bandit_threads[i] = p
    p.start()

for i in range(n_problems):
      bandit_threads[i].join()

for t in range(max_time):
    avg_reward[t] = avg_reward[t]/n_problems

plt.plot(avg_reward)
plt.ylabel("Average reward [a. u.]")
plt.show()

# k = 10
# t_stop=1000
# n=2000
#
# kmb = bandit.KMultiBandit(k)
# avg_reward_vs_t = kmb.run_multiple_simulations(t_stop=t_stop,
#                                                epsilon=0.1,
#                                                n=n)
#
# print("Reward: " + str(avg_reward_vs_t))