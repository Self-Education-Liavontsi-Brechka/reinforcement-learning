from utils import rand_choice, rand_un
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


class EpsilonGreedyAgent(BanditAgent):
    def __init__(self, number_of_actions, action_value_method, epsilon=0.0, step_size=None):
        super(EpsilonGreedyAgent, self).__init__(number_of_actions)

        self.action_value_method = action_value_method
        self.epsilon = epsilon
        self.step_size = step_size

        self.estimated_action_values = np.zeros(len(self.actions))
        self.action_selected_count = np.zeros(len(self.actions))

    def get_action(self, state):
        action = None

        if self.epsilon > 0.0 and rand_un() <= self.epsilon:
            action = rand_choice(self.actions)
        else:
            action = self.action_value_method(self.actions, self.estimated_action_values)

        self.action_selected_count[action] += 1
        return action

    def update_est_value(self, action, reward):
        update = 0.0

        if self.step_size:
            update = self.step_size * (reward - self.estimated_action_values[action])
        else:
            update = (reward - self.estimated_action_values[action]) / self.action_selected_count[action]

        self.estimated_action_values[action] += update


def greedy(actions, est):
    result = None
    max_est = None
    for a in actions:
        if not result or est[a] > max_est or (est[a] == max_est and rand_un() < 0.5):
            result = a
            max_est = est[a]

    return result
