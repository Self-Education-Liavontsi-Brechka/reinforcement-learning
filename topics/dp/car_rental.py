import numpy as np

from lib.envs.car_rental import CarRental
from topics.dp.policy_iteration import policy_improvement

# env = CarRental(max_cars_available=10, max_cars_move=3, rent_credit=5, move_cost=1, lambdas=(2, 3, 2, 1))
env = CarRental(max_cars_available=10, max_cars_move=3, max_rent_return=5,
                rent_credit=5, move_cost=1, lambdas=(2, 3, 2, 1))

policy, v = policy_improvement(env, discount_factor=0.9)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy:")
print(np.flip(
    np.reshape(np.argmax(policy, axis=1) - env.max_car_move, (env.max_cars_available + 1, env.max_cars_available + 1)),
    0))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(np.flip(v.reshape((env.max_cars_available + 1, env.max_cars_available + 1)), 0))
print("")
