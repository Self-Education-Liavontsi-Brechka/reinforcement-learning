import gym

from gym import spaces
from gym.utils import seeding


class RandomWalk1000(gym.Env):
    observation_space = spaces.Discrete(1000)
    action_space = spaces.Discrete(2)

    def __init__(self):
        self.seed()
        self.nS = RandomWalk1000.observation_space.n
        self.nA = RandomWalk1000.action_space.n

    def _step(self, action):
        step = 0
        while True:
            step = self.np_random.randint(0, 201) - 100
            if step != 0:
                break

        self.current_s += step
        if self.current_s < 0:
            return -1, -1., True, {}
        elif self.current_s > 999:
            return 1000, 1., True, {}
        else:
            return self.current_s, 0., False, {}

    def _reset(self):
        self.current_s = 499
        return self.current_s

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
