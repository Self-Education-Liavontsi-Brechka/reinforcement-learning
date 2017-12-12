from exp import RandomBanditExperiment, GreedyBanditExperiment

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    max_runs = 1000
    max_steps = 1000
    number_of_arms = 10
    step_size = None

    greedy_exp_optimistic = GreedyBanditExperiment(max_runs=max_runs, max_steps=max_steps, number_of_arms=number_of_arms,
                                                   epsilon=0.0, step_size=0.1, is_stationary=True, initial_value=5.0)
    greedy_exp_optimistic.execute()
    greedy_res_optimistic = greedy_exp_optimistic.get_results()

    epsilon_01_exp_realistic = GreedyBanditExperiment(max_runs=max_runs, max_steps=max_steps,
                                                      number_of_arms=number_of_arms, epsilon=0.1, step_size=0.1,
                                                      is_stationary=True, initial_value=0.0)
    epsilon_01_exp_realistic.execute()
    epsilon_01_res_realistic = epsilon_01_exp_realistic.get_results()

    x = np.arange(1, max_steps + 1, 1, dtype='int32')

    plt.subplot(2, 1, 1)
    lines = plt.plot(x, greedy_res_optimistic[1], 'k', x, epsilon_01_res_realistic[1], 'r')
    for line in lines:
        plt.setp(line, linewidth=0.5)
    plt.ylabel('Average reward')
    plt.xlabel('Steps')
    plt.legend(('Q1 = 5, eps = 0, alpha = 0.1', 'Q1 = 0, eps = 0.1, alpha = 0.1'), loc='lower right')

    plt.subplot(2, 1, 2)
    lines = plt.plot(x, greedy_res_optimistic[0], 'k', x, epsilon_01_res_realistic[0], 'r')
    for line in lines:
        plt.setp(line, linewidth=0.5)
    plt.ylabel('% of optimal action')
    plt.xlabel('Steps')
    plt.legend(('Q1 = 5, eps = 0, alpha = 0.1', 'Q1 = 0, eps = 0.1, alpha = 0.1'), loc='lower right')

    plt.savefig('optimistic_exp.png')
