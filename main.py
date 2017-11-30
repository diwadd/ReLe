import matplotlib.pyplot as plt

from k_armed_bandit_testbed import KArmedBanditTestbed

k = 10
n_problems = 2000
max_time = 1000

kabt = KArmedBanditTestbed()
avg_reward = kabt.run()

plt.plot(avg_reward)
plt.ylabel("Average reward [a. u.]")
plt.show()