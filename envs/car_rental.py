import numpy as np

from gym.envs.toy_text import discrete


class CarRental(discrete.DiscreteEnv):
    def __init__(self, max_cars_available=20, max_cars_move=5, max_rent_return=10,
                 rent_credit=10, move_cost=2, lambdas=(3, 4, 3, 2)):
        print('Car Rental Environment Initialization:')

        if len(lambdas) != 4:
            raise ValueError('lambdas argument must be a list/tuple of length 4')

        self.max_cars_available = max_cars_available
        self.max_car_move = max_cars_move

        POISSON_UP_BOUND = max_rent_return + 1
        poisson_probs = {l: np.zeros(POISSON_UP_BOUND) for l in lambdas}
        for l in lambdas:
            for n in range(0, POISSON_UP_BOUND):
                poisson_probs[l][n] = np.exp(-l) * np.math.pow(l, n) / np.math.factorial(n)

        nS = (max_cars_available + 1) ** 2
        nA = max_cars_move * 2 + 1
        P = {}
        isd = np.ones(nS) / nS

        for c1 in range(0, max_cars_available + 1):
            for c2 in range(0, max_cars_available + 1):
                s = c1 * (max_cars_available + 1) + c2

                P[s] = {a: [] for a in range(nA)}

                for a in range(nA):
                    move = a - max_cars_move

                    if (move < 0 and -move > c2) or (move > 0 and move > c1):
                        P[s][a] = [(1.0, s, -10000.0, False)]
                        # continue
                    else:
                        for rent_first in range(0, POISSON_UP_BOUND):
                            for rent_second in range(0, POISSON_UP_BOUND):
                                c1_prime = min(c1 - move, max_cars_available)
                                c2_prime = min(c2 + move, max_cars_available)

                                real_rent_first = min(rent_first, c1_prime)
                                real_rent_second = min(rent_second, c2_prime)

                                # reward = rent_credit * (real_rent_first + real_rent_second) - move_cost * abs(move)
                                reward = rent_credit * (real_rent_first + real_rent_second)
                                if abs(move) > 1:
                                    reward -= move_cost * (abs(move) - 1)

                                c1_prime -= real_rent_first
                                c2_prime -= real_rent_second

                                prob = poisson_probs[lambdas[0]][rent_first] * \
                                       poisson_probs[lambdas[1]][rent_second]

                                for return_first in range(0, POISSON_UP_BOUND):
                                    for return_second in range(0, POISSON_UP_BOUND):
                                        c1_prime_ = min(c1_prime + return_first, max_cars_available)
                                        c2_prime_ = min(c2_prime + return_second, max_cars_available)
                                        s_prime_ = c1_prime_ * (max_cars_available + 1) + c2_prime_
                                        reward_ = reward - (4 if c1_prime_ > 10 else 0) - (4 if c2_prime_ > 10 else 0)

                                        prob_ = prob * poisson_probs[lambdas[2]][return_first] * \
                                                poisson_probs[lambdas[3]][return_second]

                                        P[s][a].append((prob_, s_prime_, reward_, False))

                if s % 10 == 0:
                    print '.',

        super(CarRental, self).__init__(nS, nA, P, isd)

        print('Done')
