import numpy as np

from lib.envs.gridworld import GridWorld


def value_iteration(env, discount_factor=1.0, theta=0.00001):
    """
        Value Iteration Algorithm.

        Args:
            env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment. 
                env.nA is a number of actions in the environment.
            theta: We stop evaluation once our value function change is less than theta for all states.
            discount_factor: Gamma discount factor.

        Returns:
            A tuple (policy, V) of the optimal policy and the optimal value function.        
        """

    V = np.zeros(env.nS)

    while True:
        delta = -float('inf')

        for s in range(env.nS):
            v = V[s]
            max_v = -float('inf')

            for a in range(env.nA):
                current_v = 0.0
                for prob, s_prime, reward, is_done in env.P[s][a]:
                    current_v += prob * (reward + discount_factor * V[s_prime])
                max_v = max(max_v, current_v)

            V[s] = max_v
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    policy = np.zeros((env.nS, env.nA))

    for s in range(env.nS):
        count = 0
        prob_a = []
        for a in range(env.nA):
            current_v = 0.0
            for prob, s_prime, reward, is_done in env.P[s][a]:
                current_v += prob * (reward + discount_factor * V[s_prime])

            if np.isclose([V[s]], [current_v])[0]:
                count += 1
                prob_a.append(1.0)
            else:
                prob_a.append(0.0)

        policy[s] = np.array(prob_a) / count

    return policy, V


if __name__ == '__main__':
    env = GridWorld()

    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
