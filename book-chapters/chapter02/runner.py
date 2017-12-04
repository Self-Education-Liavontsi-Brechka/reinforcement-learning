from exp import RandomBanditExperiment, GreedyBanditExperiment

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # random_exp = RandomBanditExperiment()
    # random_exp.execute()
    # random_exp.handle_results()

    max_runs = 2000
    max_steps = 10000
    number_of_arms = 10
    step_size = None

    # greedy_exp = GreedyBanditExperiment(max_runs, max_steps, number_of_arms, 0.0, step_size)
    # greedy_exp.execute()
    # greedy_res = greedy_exp.get_results()
    #
    # epsilon_01_exp = GreedyBanditExperiment(max_runs, max_steps, number_of_arms, 0.1, step_size)
    # epsilon_01_exp.execute()
    # epsilon_01_res = epsilon_01_exp.get_results()
    #
    # epsilon_001_exp = GreedyBanditExperiment(max_runs, max_steps, number_of_arms, 0.01, step_size)
    # epsilon_001_exp.execute()
    # epsilon_001_res = epsilon_001_exp.get_results()

    non_stat_exp = GreedyBanditExperiment(max_runs, max_steps, number_of_arms, 0.1, None, False)
    non_stat_exp.execute()
    non_stat_exp_res = non_stat_exp.get_results()

    non_stat_exp_with_step = GreedyBanditExperiment(max_runs, max_steps, number_of_arms, 0.1, 0.1, False)
    non_stat_exp_with_step.execute()
    non_stat_exp_with_step_res = non_stat_exp_with_step.get_results()

    x = np.arange(1, max_steps + 1, 1, dtype='int32')

    plt.subplot(2, 1, 1)
    # lines = plt.plot(x, greedy_res[1], 'k', x, epsilon_01_res[1], 'r', x, epsilon_001_res[1], 'g')
    lines = plt.plot(x, non_stat_exp_res[1], 'k', x, non_stat_exp_with_step_res[1], 'r')
    plt.setp(lines[0], linewidth=0.5)
    plt.setp(lines[1], linewidth=0.5)
    # plt.setp(lines[2], linewidth=0.5)
    plt.ylabel('Average reward')
    plt.xlabel('Steps')
    plt.legend(('alpha = 1 / n', 'alpha = 0.1'), loc='upper right')

    plt.subplot(2, 1, 2)
    # lines = plt.plot(x, greedy_res[0] * 100.0, 'k', x, epsilon_01_res[0] * 100.0, 'r', x, epsilon_001_res[0] * 100.0, 'g')
    lines = plt.plot(x, non_stat_exp_res[0], 'k', x, non_stat_exp_with_step_res[0], 'r')
    plt.setp(lines[0], linewidth=0.5)
    plt.setp(lines[1], linewidth=0.5)
    # plt.setp(lines[2], linewidth=0.5)
    plt.ylabel('% of optimal action')
    plt.xlabel('Steps')
    plt.legend(('alpha = 1 / n', 'alpha = 0.1'), loc='upper right')

    # plt.savefig('epsilon_exp.png')
    plt.savefig('non_stat_exp.png')
