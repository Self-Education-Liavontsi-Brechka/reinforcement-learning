from collections import defaultdict

import gym
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import pandas as pd

ENV = gym.make('CartPole-v0')

STATE_BOUNDS = list(zip(ENV.observation_space.low, ENV.observation_space.high))
STATE_BOUNDS[1] = [-.5, .5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]

NUM_BUCKETS = (1, 1, 6, 3)

MIN_ALPHA = 0.1
MIN_EPSILON = 0.01

NUM_EPISODES = 400
MAX_T = 200

Q = defaultdict(lambda: np.zeros(ENV.action_space.n))


def run():
    alpha = get_alpha(0)
    epsilon = get_alpha(0)
    discount_factor = 0.99

    episode_rewards = []

    for i_episode in xrange(NUM_EPISODES):
        episode_reward = 0.0

        s = adjust_state(ENV.reset())

        for _ in xrange(MAX_T):
            ENV.render()

            a_probs = epsilon_greedy_policy(s, epsilon)
            a = np.random.choice(np.arange(len(a_probs)), p=a_probs)
            s_prime, reward, done, info = ENV.step(a)
            s_prime = adjust_state(s_prime)

            Q[s][a] += alpha * (reward + discount_factor * np.amax(Q[s_prime]) - Q[s][a])

            episode_reward += reward

            if done:
                print 'Episode #{0} is finished with reward {1}'.format(i_episode + 1, episode_reward)
                episode_rewards.append(episode_reward)
                break

            s = s_prime

        alpha = get_alpha(i_episode)
        epsilon = get_epsilon(i_episode)

    plot_episode_stats(episode_rewards)


def adjust_state(state):
    adjusted_state = []
    for i in xrange(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        adjusted_state.append(bucket_index)
    return tuple(adjusted_state)


def epsilon_greedy_policy(s, epsilon):
    A = np.ones(ENV.action_space.n, dtype=float) * epsilon / ENV.action_space.n
    best_action = np.argmax(Q[s])
    A[best_action] += (1.0 - epsilon)
    return A


def get_epsilon(t):
    return max(MIN_EPSILON, min(1.0, 1.0 - math.log10((t + 1.0) / 25.0)))


def get_alpha(t):
    return max(MIN_ALPHA, min(0.5, 1.0 - math.log10((t + 1.0) / 25.0)))


def plot_episode_stats(episode_rewards, smoothing_window=10):
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show()


if __name__ == '__main__':
    run()
