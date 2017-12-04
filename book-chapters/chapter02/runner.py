from exp import RandomBanditExperiment, GreedyBanditExperiment
from agent import greedy

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # random_exp = RandomBanditExperiment()
    # random_exp.execute()
    # random_exp.handle_results()

    action_value_method = greedy
    max_runs = 2000
    max_steps = 10000
    number_of_arms = 10
    step_size = None

    greedy_exp = GreedyBanditExperiment(action_value_method, max_runs, max_steps, number_of_arms, 0.0, step_size)
    greedy_exp.execute()
    greedy_res = greedy_exp.get_results()

    epsilon_01_exp = GreedyBanditExperiment(action_value_method, max_runs, max_steps, number_of_arms, 0.1, step_size)
    epsilon_01_exp.execute()
    epsilon_01_res = epsilon_01_exp.get_results()

    epsilon_001_exp = GreedyBanditExperiment(action_value_method, max_runs, max_steps, number_of_arms, 0.01, step_size)
    epsilon_001_exp.execute()
    epsilon_001_res = epsilon_001_exp.get_results()

    x = np.arange(1, max_steps + 1, 1, dtype='int32')

    plt.subplot(2, 1, 1)
    lines = plt.plot(x, greedy_res[1], 'k', x, epsilon_01_res[1], 'r', x, epsilon_001_res[1], 'g')
    plt.setp(lines[0], linewidth=0.5)
    plt.setp(lines[1], linewidth=0.5)
    plt.setp(lines[2], linewidth=0.5)
    plt.ylabel('Average reward')
    plt.xlabel('Steps')
    plt.legend(('Greedy', 'Epsilon = 0.1', 'Epsilon = 0.01'), loc='upper right')

    plt.subplot(2, 1, 2)
    lines = plt.plot(x, greedy_res[0] * 100.0, 'k', x, epsilon_01_res[0] * 100.0, 'r', x, epsilon_001_res[0] * 100.0, 'g')
    plt.setp(lines[0], linewidth=0.5)
    plt.setp(lines[1], linewidth=0.5)
    plt.setp(lines[2], linewidth=0.5)
    plt.ylabel('% of optimal action')
    plt.xlabel('Steps')
    plt.legend(('Greedy', 'Epsilon = 0.1', 'Epsilon = 0.01'), loc='upper right')

    plt.savefig('epsilon_exp.png')
