import gym

from gym import spaces
from gym.utils import seeding


class RandomWalk1000(gym.Env):
    observation_space = spaces.Discrete(1000)
    action_space = [-1, 1]

    def __init__(self):
        self.seed()

    def _step(self, action):
        if action not in RandomWalk1000.action_space:
            raise RuntimeError("Wrong action")

        self.current_s += self.np_random.randint(1, 101) * action
        if self.current_s <= 0:
            return 0, -1., True, {}
        elif self.current_s >= 1001:
            return 1001, 1., True, {}
        else:
            return self.current_s, 0., False, {}

    def _reset(self):
        # self.current_s = int(self.np_random.normal(500.0, 100.0))
        self.current_s = 500
        return self.current_s

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
