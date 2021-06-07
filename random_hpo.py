import math
import numpy as np
from model import PARAMETER_NAMES

def random_hyperparameters(parameters_range, n, seed=0):
    np.random.seed(seed)
    parameters = dict()

    for param_name, (low, high, distribution) in parameters_range.items():
        if distribution == 'discrete':
            parameters[param_name] = generate_discrete_uniform(low, high, n)
        elif distribution == 'uniform':
            parameters[param_name] = generate_uniform(low, high, n)
        elif distribution[:10] == 'loguniform':
            parameters[param_name] = generate_loguniform(low, high, n, int(distribution[11:]))
        elif distribution[:19] == 'discrete_loguniform':
            parameters[param_name] = generate_discrete_loguniform(low, high, n, int(distribution[20:]))
        elif distribution == 'fixed':
            if low != high:
                raise ValueError
            parameters[param_name] = [low]*n
        else:
            raise ValueError

    return parameters


def generate_discrete_uniform(low, high, n):
    return np.random.random_integers(low, high, (n,))


def generate_uniform(low, high, n):
    return np.random.uniform(low, high, (n,))


def generate_loguniform(low, high, n, base):
    return np.power(base, generate_uniform(math.log(low, base), math.log(high, base), n))


def generate_discrete_loguniform(low, high, n, base):
    return np.power(base, generate_discrete_uniform(np.floor(math.log(low, base)), np.floor(math.log(high, base)), n).astype(float)).astype(int)
