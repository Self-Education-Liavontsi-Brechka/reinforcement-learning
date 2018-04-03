import numpy as np
import matplotlib.pyplot as plt

from lib.envs.random_walk_1000 import RandomWalk1000

episodes = 100000

env = RandomWalk1000()
V = np.zeros(1000)

for i_episode in range(episodes):
    s = env.reset()
    episode = [s]

    while True:
        s_prime, reward, done, _ = env.step(np.random.choice([-1, 1]))

        if done:
            for i, s in enumerate(episode):
                V[s] += reward
            break

        episode.append(s_prime - 1)

    if (i_episode + 1) % (episodes / 10) == 0:
        print '.',

V /= episodes

plt.plot(np.arange(1, 1001), V)
plt.show()
