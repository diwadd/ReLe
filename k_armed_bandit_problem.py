import threading
import logging
import time
import random

import k_armed_bandit_class as kabc


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s')

class KArmedBanditProblem(threading.Thread):


    def __init__(self,
                 group=None,
                 target=None,
                 name=None,
                 args=(),
                 kwargs={},
                 daemon=None):

        threading.Thread.__init__(self,
                                  group=group,
                                  target=target,
                                  name=name,
                                  daemon=daemon)

        self.k = kwargs["k"]


    def solve(self):
        logging.debug("Solving for k = {0}".format(self.k))
        time.sleep(random.randint(0, 5))

    def run(self):
        logging.debug("{0}".format(threading.current_thread()))
        self.solve()
        logging.debug("Ending...")
        return