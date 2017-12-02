from utils import rand_choice
import numpy as np


class BanditAgent(object):
    def __init__(self, number_of_actions):
        self.actions = np.arange(number_of_actions, dtype="int64")

    def get_action(self, state):
        pass

    def update_est_value(self, action, reward):
        pass


class RandomAgent(BanditAgent):
    def __init__(self, number_of_actions):
        super(RandomAgent, self).__init__(number_of_actions)

    def get_action(self, state):
        return rand_choice(self.actions)