from environment import KArmedTestbed
from agent import RandomAgent, EpsilonGreedyAgent
from utils import plot_results

import numpy as np
import sys


class BanditExperiment(object):
    def __init__(self, max_runs=2000, max_steps=1000, number_of_arms=10):
        self.max_runs = max_runs
        self.max_steps = max_steps
        self.number_of_arms = number_of_arms

        self.optimality = np.full(max_steps, 0.0)
        self.avg_reward = np.full(max_steps, 0.0)

    def execute(self):
        pass

    def get_results(self):
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

            for i in np.arange(self.max_steps):
                current_state = environment.get_current_state()
                next_action = agent.get_action(current_state)
                reward = environment.do_action(next_action)

                if not self.avg_reward[i]:
                    self.avg_reward[i] = reward
                else:
                    self.avg_reward[i] = self.avg_reward[i] + (reward - self.avg_reward[i]) / (i + 1.0)

    def handle_results(self):
        plot_results(np.arange(1, self.max_steps + 1, 1), self.avg_reward, 'random_exp.png')


class GreedyBanditExperiment(BanditExperiment):
    def __init__(self, action_value_method, max_runs=2000, max_steps=1000, number_of_arms=10,
                 epsilon=0.0, step_size=None):
        super(GreedyBanditExperiment, self).__init__(max_runs, max_steps, number_of_arms)

        self.action_value_method = action_value_method
        self.epsilon = epsilon
        self.step_size = step_size

    def execute(self):
        for k in np.arange(self.max_runs):
            environment = KArmedTestbed(self.number_of_arms)
            agent = EpsilonGreedyAgent(self.number_of_arms, self.action_value_method, self.epsilon, self.step_size)

            for i in np.arange(self.max_steps):
                current_state = environment.get_current_state()
                next_action = agent.get_action(current_state)
                reward = environment.do_action(next_action)
                agent.update_est_value(next_action, reward)

                self.avg_reward[i] += reward

            if k % 100 == 0:
                print '.',
                sys.stdout.flush()

        self.avg_reward /= self.max_runs
        print 'Done'

    def get_results(self):
        return self.avg_reward

    def handle_results(self):
        plot_results(np.arange(1, self.max_steps + 1, 1), self.avg_reward, 'greedy_exp_%d_.png' % self.epsilon)
