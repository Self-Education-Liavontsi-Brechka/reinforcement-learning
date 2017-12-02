from environment import KArmedTestbed
from agent import RandomAgent
from utils import plot_results

import numpy as np


class BanditExperiment(object):
    def __init__(self, max_runs=2000, max_steps=1000, number_of_arms=10):
        self.max_runs = max_runs
        self.max_steps = max_steps
        self.number_of_arms = number_of_arms

        self.optimality = np.full(max_steps, None)
        self.avg_reward = np.full(max_steps, None)

    def execute(self):
        pass

    def handle_results(self):
        pass


class RandomBanditExperiment(BanditExperiment):
    def __init__(self, max_runs=2000, max_steps=1000, number_of_arms=10):
        super(RandomBanditExperiment, self).__init__(max_runs, max_steps, number_of_arms)

    def execute(self):
        for k in np.arange(self.max_runs):
            environment = KArmedTestbed(self.number_of_arms)
            agent = RandomAgent(self.number_of_arms)

            num_steps = 0

            for i in np.arange(self.max_steps):
                current_state = environment.get_current_state()
                next_action = agent.get_action(current_state)
                reward = environment.do_action(next_action)

                num_steps += 1
                if not self.avg_reward[i]:
                    self.avg_reward[i] = reward
                else:
                    self.avg_reward[i] = self.avg_reward[i] + (reward - self.avg_reward[i]) / num_steps

    def handle_results(self):
        plot_results(np.arange(1, self.max_steps + 1, 1), self.avg_reward, 'random_exp.png')
