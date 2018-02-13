import numpy as np

from sys import maxint
from lib.envs.gridworld import GridWorld
from policy_evaluation import policy_eval


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        is_stable = True

        for s in range(env.nS):
            old_pi_a = np.copy(policy[s])
            max_v = -maxint

            for a in range(env.nA):
                current_v = 0.0

                for prob, s_prime, reward, done in env.P[s][a]:
                    current_v += prob * (reward + discount_factor * V[s_prime])

                max_v = max(max_v, current_v)

            count = 0
            prob_a = []
            for a in range(env.nA):
                current_v = 0.0

                for prob, s_prime, reward, done in env.P[s][a]:
                    current_v += prob * (reward + discount_factor * V[s_prime])

                if np.isclose([max_v], [current_v])[0]:
                    count += 1
                    prob_a.append(1.0)
                else:
                    prob_a.append(0.0)

            policy[s] = np.array(prob_a) / count
            is_stable = is_stable and np.all(np.isclose(old_pi_a, policy[s]))

        if is_stable:
            break

    return policy, V


if __name__ == '__main__':
    env = GridWorld()

    policy, v = policy_improvement(env)
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
