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
    def __init__(self, number_of_actions, epsilon=0.0, step_size=None, initial_value=0.0):
        super(EpsilonGreedyAgent, self).__init__(number_of_actions)

        self.epsilon = epsilon
        self.step_size = step_size

        self.estimated_action_values = np.full(len(self.actions), initial_value)
        self.action_selected_count = np.zeros(len(self.actions))

    def get_action(self, state):
        action = None

        if self.epsilon > 0.0 and np.random.binomial(1, self.epsilon) == 1:
            action = rand_choice(self.actions)
        else:
            action = np.argmax(self.estimated_action_values)

        self.action_selected_count[action] += 1
        return action

    def update_est_value(self, action, reward):
        update = 0.0

        if self.step_size:
            update = self.step_size * (reward - self.estimated_action_values[action])
        else:
            update = (reward - self.estimated_action_values[action]) / self.action_selected_count[action]

        self.estimated_action_values[action] += update
