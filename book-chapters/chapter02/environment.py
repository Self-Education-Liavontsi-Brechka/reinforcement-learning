from utils import rand_norm
import numpy as np


class Environment(object):
    def __init__(self):
        pass

    def get_current_state(self):
        pass

    # def get_possible_actions(self, state):
    #     pass

    def do_action(self, action):
        pass

    def reset(self):
        pass

    def is_terminated(self):
        pass


class KArmedTestbed(Environment):
    def __init__(self, number_of_arms):
        super(KArmedTestbed, self).__init__()

        self.number_of_arms = number_of_arms
        self.action_value = np.array([rand_norm(0, 1.0) for i in np.arange(number_of_arms)])
        self.current_state = np.zeros(0)

    def get_current_state(self):
        return self.current_state

    def do_action(self, action):
        # state update goes here
        return rand_norm(self.action_value[action], 1.0)

    def is_terminated(self):
        return False

