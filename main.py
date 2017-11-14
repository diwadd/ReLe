import bandit

k = 10
t_stop=100
n=2000

kmb = bandit.KMultiBandit(k)
avg_reward_vs_t = kmb.run_multiple_simulations(t_stop=t_stop,
                                               epsilon=0.1,
                                               n=n)

print("Reward: " + str(avg_reward_vs_t))