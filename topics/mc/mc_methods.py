import numpy as np
from collections import defaultdict
from common.policy import create_greedy_policy, make_epsilon_greedy_policy


def mc_on_policy_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    # The final value function
    V = defaultdict(float)

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for _ in xrange(num_episodes):
        s = env.reset()

        episode = []
        while True:
            a = policy(s)
            s_prime, reward, done, info = env.step(a)
            episode.append((s, reward))

            if done:
                break

            s = s_prime

        state_set = set([step[0] for step in episode])
        for s in state_set:
            first_i = next(i for i, step in enumerate(episode) if step[0] == s)
            G = sum(step[1] * (discount_factor ** i) for i, step in enumerate(episode[first_i:]))

            returns_sum[s] += G
            returns_count[s] += 1.0
            V[s] = returns_sum[s] / returns_count[s]

    return V


def mc_on_policy_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for _ in xrange(num_episodes):
        episode = []

        s = env.reset()
        while True:
            a_probs = policy(s)
            a = np.random.choice(np.arange(len(a_probs)), p=a_probs)
            s_prime, reward, done, info = env.step(a)

            episode.append((s, a, reward))

            if done:
                break

            s = s_prime

        state_action_set = set([(s, a) for s, a, reward in episode])
        for (s, a) in state_action_set:
            first_i = next(i for i, episode_step in enumerate(episode) if episode_step[0] == s and episode_step[1] == a)
            G = sum(episode_step[2] * (discount_factor ** i) for i, episode_step in enumerate(episode[first_i:]))

            pair = (s, a)
            returns_sum[pair] += G
            returns_count[pair] += 1.0
            Q[s][a] = returns_sum[pair] / returns_count[pair]

    return Q, policy


def mc_off_policy_control_weighted(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
        Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
        Finds an optimal greedy policy.

        Args:
            env: OpenAI gym environment.
            num_episodes: Number of episodes to sample.
            behavior_policy: The behavior to follow while generating episodes.
                A function that given an observation returns a vector of probabilities for each action.
            discount_factor: Gamma discount factor.

        Returns:
            A tuple (Q, policy).
            Q is a dictionary mapping state -> action values.
            policy is a function that takes an observation as an argument and returns
            action probabilities. This is the optimal greedy policy.
        """

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)

    for _ in xrange(num_episodes):
        episode = []
        s = env.reset()
        while True:
            a_probs = behavior_policy(s)
            a = np.random.choice(np.arange(len(a_probs)), p=a_probs)
            s_prime, reward, done, info = env.step(a)
            episode.append((s, a, reward))

            if done:
                break

            s = s_prime

        G = 0
        W = 1
        for s, a, reward in reversed(episode):
            G = discount_factor * G + reward
            C[s][a] += W
            Q[s][a] += W / C[s][a] * (G - Q[s][a])

            if a != np.argmax(target_policy(s)):
                break

            W *= 1.0 / behavior_policy(s)[a]

    return Q, target_policy
