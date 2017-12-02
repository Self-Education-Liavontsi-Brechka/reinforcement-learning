#!/usr/bin/env python
import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt


def rand_in_range(max):  # returns integer, max: integer
    return rnd.randint(max + 1)


def rand_un():  # returns floating point
    return rnd.uniform()


def rand_norm(mu, sigma):  # returns floating point, mu: floating point, sigma: floating point
    return rnd.normal(mu, sigma)


def rand_choice(elements):
    return rnd.choice(elements)


def save_results(data, filename):
    with open(filename, 'w') as data_file:
        for i in np.arange(len(data)):
            data_file.write('{0}\n'.format(data[i]))


def plot_results(x, y, filename):
    plt.plot(x, y)
    plt.savefig(filename)
