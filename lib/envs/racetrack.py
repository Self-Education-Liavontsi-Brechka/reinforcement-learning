# coding=utf-8
import gym
import numpy as np

from gym import spaces


class RacetrackEnv(gym.Env):
    """
    Racetrack env for ex. 5.8 in Sutton, Barto, 2017

    Consider driving a race car around a turn like those
    shown in Figure 5.5. You want to go as fast as possible, but not so fast as to run o↵ the track. In our
    simplified racetrack, the car is at one of a discrete set of grid positions, the cells in the diagram. The
    velocity is also discrete, a number of grid cells moved horizontally and vertically per time step. The
    actions are increments to the velocity components. Each may be changed by +1, −1, or 0 in one step,
    for a total of nine actions. Both velocity components are restricted to be nonnegative and less than 5,
    and they cannot both be zero except at the starting line. Each episode begins in one of the randomly
    selected start states with both velocity components zero and ends when the car crosses the finish line.
    The rewards are −1 for each step until the car crosses the finish line. If the car hits the track boundary,
    it is moved back to a random position on the starting line, both velocity components are reduced to
    zero, and the episode continues. Before updating the car’s location at each time step, check to see if
    the projected path of the car intersects the track boundary. If it intersects the finish line, the episode
    ends; if it intersects anywhere else, the car is considered to have hit the track boundary and is sent
    back to the starting line. To make the task more challenging, with probability 0.1 at each time step
    the velocity increments are both zero, independently of the intended increments. Apply a Monte Carlo
    control method to this task to compute the optimal policy from each starting state. Exhibit several
    trajectories following the optimal policy (but turn the noise o↵ for these trajectories).

    Takes an array of strings as a parameter where (the starting position is in the left top corner):
        # - non-reachable cell
        _ - start cell
        . - empty cell
        * - finish cell

    An action space is represented as a discrete space where:
        v * 3 + h = action, where v and h are vertical and horizontal velocity changes respectively

    An observation space is represented as a tuple of 4 Discrete spaces where:
        1st space - row coordinate
        2nd - column coordinate
        3rd - vertical velocity
        4th - horizontal velocity
    """
    NON_REACHABLE_CELL, START_CELL, EMPTY_CELL, FINISH_CELL = '#', '_', '.', '*'

    def __init__(self, track_grid):
        assert len(track_grid) >= 1
        assert len(set([len(row) for row in track_grid])) == 1

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Tuple((spaces.Discrete(len(track_grid)),
                                               spaces.Discrete(len(track_grid[0])),
                                               spaces.Discrete(5),
                                               spaces.Discrete(5)))

        self.track_grid = []
        self.track_grid.append(RacetrackEnv.NON_REACHABLE_CELL * len(track_grid[0]))
        for row in track_grid:
            self.track_grid.append(''.join([RacetrackEnv.NON_REACHABLE_CELL, row, RacetrackEnv.NON_REACHABLE_CELL]))
        self.track_grid.append(RacetrackEnv.NON_REACHABLE_CELL * len(track_grid[0]))

        self.start_position = [
            (row, column)
            for row, cells in enumerate(self.track_grid) for column, cell in enumerate(cells)
            if cell == RacetrackEnv.START_CELL
        ]

    def _step(self, raw_action):
        assert 0 <= raw_action < 9

        action = (raw_action // 3 - 1, raw_action % 3 - 1)
        assert -1 <= action[0] <= 1 and -1 <= action[1] <= action[1]

        action = (0, 0) if np.random.choice(2, p=[0.1, 0.9]) == 0 else action

        velocities = self.velocities[0] + action[0], self.velocities[1] + action[1]
        if (velocities == (0, 0) and
            self.track_grid[self.current_position[0]][self.current_position[1]] != RacetrackEnv.START_CELL) or \
                velocities[0] > 4 or velocities[0] < 0 or velocities[1] > 4 or velocities[1] < 0:
            return (self.current_position[0], self.current_position[1], self.velocities[0], self.velocities[1]), \
                   -1, False, {}

        position = \
            max(0, self.current_position[0] - velocities[0]), \
            min(len(self.track_grid[0]), self.current_position[1] + velocities[1])

        # TODO: should be changed to handle hitting the track boundary properly
        if next((cell
                 for cells in self.track_grid[position[0]:self.current_position[0] + 1]
                 for cell in cells[self.current_position[1]:position[1] + 1]
                 if cell == RacetrackEnv.FINISH_CELL), None):
            return (position[0], position[1], velocities[0], velocities[1]), 0, True, {}

        if next((cell
                 for cells in self.track_grid[position[0]:self.current_position[0] + 1]
                 for cell in cells[self.current_position[1]:position[1] + 1]
                 if cell == RacetrackEnv.NON_REACHABLE_CELL), None):
            return self.reset(), -1, False, {}

        self.velocities = velocities
        self.current_position = position

        return (position[0], position[1], velocities[0], velocities[1]), -1, False, {}

    def _reset(self):
        self.velocities = (0, 0)
        self.current_position = self.start_position[np.random.choice(len(self.start_position))]

        return self.current_position[0], self.current_position[1], self.velocities[0], self.velocities[1]
