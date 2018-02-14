import numpy as np
import sys

from collections import defaultdict
from lib import plotting


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print "\rEpisode {0}/{1}.".format(i_episode + 1, num_episodes)
            sys.stdout.flush()

        s = env.reset()
        a_probs = policy(s)
        a = np.random.choice(np.arange(len(a_probs)), p=a_probs)

        while True:
            s_prime, reward, done, info = env.step(a)
            a_prime_probs = policy(s_prime)
            a_prime = np.random.choice(np.arange(len(a_prime_probs)), p=a_prime_probs)

            Q[s][a] += alpha * (reward + discount_factor * Q[s_prime][a_prime] - Q[s][a])

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] += 1

            if done:
                break

            s = s_prime
            a = a_prime

    return Q, stats


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print "\rEpisode {0}/{1}.".format(i_episode + 1, num_episodes)
            sys.stdout.flush()

        s = env.reset()
        while True:
            a_probs = policy(s)
            a = np.random.choice(np.arange(len(a_probs)), p=a_probs)
            s_prime, reward, done, info = env.step(a)

            Q[s][a] += alpha * (reward + discount_factor * max(Q[s_prime]) - Q[s][a])

            stats.episode_lengths[i_episode] += 1
            stats.episode_rewards[i_episode] += reward

            if done:
                break

            s = s_prime

    return Q, stats


def expected_sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print "\rEpisode {0}/{1}.".format(i_episode + 1, num_episodes)
            sys.stdout.flush()

        s = env.reset()
        while True:
            a_probs = policy(s)
            a = np.random.choice(np.arange(len(a_probs)), p=a_probs)
            s_prime, reward, done, info = env.step(a)
            a_prime_probs = policy(s_prime)

            expected_value = 0.0
            for a_prime, a_prime_prob in enumerate(a_prime_probs):
                expected_value += a_prime_prob * Q[s_prime][a_prime]

            Q[s][a] += alpha * (reward + discount_factor * expected_value - Q[s][a])

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] += 1

            if done:
                break

            s = s_prime

    return Q, stats
