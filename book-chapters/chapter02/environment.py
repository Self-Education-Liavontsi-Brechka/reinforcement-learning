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
        self.available_actions = np.array([i for i in np.arange(1, number_of_arms + 1, 1)])

    def get_current_state(self):
        return self.current_state

    def do_action(self, action):
        # state update goes here
        return rand_norm(self.action_value[action], 1.0)

    def is_terminated(self):
        return False


class KArmedTestbedNonStationary(KArmedTestbed):
    def __init__(self, number_of_arms):
        super(KArmedTestbedNonStationary, self).__init__(number_of_arms)

        self.action_value = np.ones(number_of_arms)

    def do_action(self, action):
        for a in self.available_actions:
            self.action_value[a - 1] += rand_norm(0.0, 0.01)

        return rand_norm(self.action_value[action], 1.0)
