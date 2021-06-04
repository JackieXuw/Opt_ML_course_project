import numpy as np

PARAMETER_NAMES = {'lr', 'batch_size', 'epochs', 'hidden', 'layers'}


def random_hyperparameters(parameters_range, n):
    parameters = dict()

    for param_name, (low, high, distribution) in parameters_range.items():
        if distribution == 'discrete':
            parameters[param_name] = generate_discrete_uniform(low, high, n)
        elif distribution == 'uniform':
            parameters[param_name] = generate_uniform(low, high, n)
        elif distribution == 'loguniform':
            parameters[param_name] = generate_loguniform(low, high, n)
        elif distribution == 'discrete_loguniform':
            parameters[param_name] = generate_discrete_loguniform(low, high, n)
        elif distribution == 'fixed':
            if low != high:
                raise ValueError
            parameters[param_name] = [low]*n

    if parameters.keys() != PARAMETER_NAMES:
        raise ValueError

    return parameters


def generate_discrete_uniform(low, high, n):
    return np.random.random_integers(low, high, (n,))


def generate_uniform(low, high, n):
    return np.random.uniform(low, high, (n,))


def generate_loguniform(low, high, n, base=2):
    return np.power(base, generate_uniform(low, high, n))


def generate_discrete_loguniform(low, high, n, base=2):
    return np.power(base, generate_discrete_uniform(low, high, n))


a = random_hyperparameters({'lr': (0, 1, 'uniform'),
                        'batch_size': (0, 4, 'discrete'),
                        'hidden': (8, 12, 'discrete_loguniform'),
                        'layers': (1, 1, 'fixed'),
                        'epochs': (100, 100, 'fixed')}, 10)
