import bandit

import k_armed_bandit_problem as kabp

k = 10
n_problems = 10

for i in range(n_problems):
    p = kabp.KArmedBanditProblem(kwargs={"k": k})
    p.start()



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