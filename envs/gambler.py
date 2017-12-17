import numpy as np

from gym.envs.toy_text import discrete


class GamblerEnv(discrete.DiscreteEnv):
    def __init__(self, goal_amount=100, head_prob=0.4):
        nS = goal_amount + 1
        nA = goal_amount // 2 + 1
        P = dict()
        isd = np.ones(nS) / nS

        for s in range(0, nS, 1):
            P[s] = {a: [] for a in range(nA)}
            if s == 0 or s == goal_amount:
                continue

            for a in range(nA):
                if a > s or s + a > goal_amount:
                    continue

                P[s][a].append((head_prob, s + a, 1 if s + a == goal_amount else 0, s + a == goal_amount))
                P[s][a].append((1 - head_prob, s - a, 1 if s - a == goal_amount else 0, s - a == goal_amount))

        super(GamblerEnv, self).__init__(nS, nA, P, isd)
