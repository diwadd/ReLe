import random


class KArmedBandit:

    def __init__(self,
                 mu=0.0,
                 var=0.0):

        self.action_value = random.gauss(mu, var)


    def reward(self,
               var=1.0):
        r_t = random.gauss(self.action_value, var)
        return r_t

    def __repr__(self):
        r = "%r, action_value: %s".format(self.__class__, self.action_value)
        return r

    def __str__(self):
        s = "KArmedBandit, action_value: %s".format(self.action_value)
        return s