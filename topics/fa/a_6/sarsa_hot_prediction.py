import numpy as np
import matplotlib.pyplot as plt

from lib.envs.random_walk_1000 import RandomWalk1000

env = RandomWalk1000()


def observation_encoder(observation):
    assert env.observation_space.contains(observation)
    features = np.zeros((env.nS, 1), dtype=np.int32)
    features[observation][0] = 1
    return features


def value_estimator(observation, weights):
    return np.dot(weights.transpose(), observation_encoder(observation))


alpha = 0.5
discount = 1.0
episodes = 10000
weights = np.zeros((env.nS, 1), dtype=np.float32)

for i_episode in range(episodes):
    s = env.reset()

    while True:
        s_prime, reward, done, _ = env.step(None)

        if done:
            weights += alpha * (reward - value_estimator(s, weights)) * observation_encoder(s)
            break

        weights += alpha * (reward + discount * value_estimator(s_prime, weights) - value_estimator(s, weights)) * \
                   observation_encoder(s)

        s = s_prime

    if (i_episode + 1) % (episodes / 10) == 0:
        print '.',

V = np.zeros(env.observation_space.n, dtype=np.float32)
for s in range(env.observation_space.n):
    V[s] = value_estimator(s, weights)

plt.plot(np.arange(env.observation_space.n), V)
plt.show()
