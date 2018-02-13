import numpy as np
import matplotlib.pyplot as plt

from lib.envs.gambler import GamblerEnv
from value_iteration import value_iteration

env = GamblerEnv(goal_amount=100)

policy, v = value_iteration(env)
v[env.nS - 1] = 1.0

greedy_policy = np.zeros(env.nS)
for s in range(1, env.nS - 1):
    max_prob = -float('inf')
    max_a = -1
    for a in range(env.nA):
        if policy[s][a] >= max_prob:
            max_prob = policy[s][a]
            max_a = a
    greedy_policy[s] = max_a

plt.figure(1)
plt.xlabel('Capital')
plt.ylabel('Value Estimates')
plt.plot(v)

plt.figure(2)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.scatter(range(env.nS), greedy_policy)
plt.show()
