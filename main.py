import matplotlib.pyplot as plt

from k_armed_bandit_testbed import KArmedBanditTestbed

k = 10
n_problems = 2000
max_time = 1000

avg_reward_0p0, optimal_action_percent_0p0 = KArmedBanditTestbed(kwargs={"epsilon": 0.0}).run()
avg_reward_0p1, optimal_action_percent_0p1 = KArmedBanditTestbed(kwargs={"epsilon": 0.1}).run()
avg_reward_0p01, optimal_action_percent_0p01 = KArmedBanditTestbed(kwargs={"epsilon": 0.01}).run()

plt.plot(avg_reward_0p0, 'g')
plt.plot(avg_reward_0p01, 'r')
plt.plot(avg_reward_0p1, 'k')
plt.ylabel("Average reward [a. u.]")
plt.show()

plt.plot(optimal_action_percent_0p0, 'g')
plt.plot(optimal_action_percent_0p01, 'r')
plt.plot(optimal_action_percent_0p1, 'k')
plt.ylabel("Optimal action [%]")
plt.ylim(0, 110)
plt.show()