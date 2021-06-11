import math
import numpy as np
from model import *
from torch import optim


def random_hyperparameters(parameters_range, n, seed=0):
    np.random.seed(seed)
    parameters = dict()

    for name, (low, high, distribution) in parameters_range.items():
        if distribution == 'discrete':
            parameters[name] = generate_discrete_uniform(low, high, n)
        elif distribution == 'uniform':
            parameters[name] = generate_uniform(low, high, n)
        elif distribution[:10] == 'loguniform':
            parameters[name] = generate_loguniform(low, high, n, int(distribution[11:]))
        elif distribution[:19] == 'discrete_loguniform':
            parameters[name] = generate_discrete_loguniform(low, high, n, int(distribution[20:]))
        elif distribution == 'fixed' and low == high:
            parameters[name] = [low]*n
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


def random_search(parameters):
    p = dict()
    #results = dict()
    results = list()

    num_trials = len(parameters['num_hidden'])
    for i in range(num_trials):
        for k in parameters.keys():
            p[k] = parameters[k][i]

        model = Net(num_hidden=p['num_hidden'], num_layers=p['num_layers'])
        sgd = optim.SGD(model.parameters(), lr=p['lr'], momentum=p['momentum'])
        train_error, test_error, exec_time = run(model, sgd, mini_batch_size=p['mini_batch_size'], num_epochs=p['num_epochs'])
        result = dict(zip(p.keys(), p.values()))
        result['train_error'] = train_error
        result['test_error'] = test_error
        result['exec_time'] = exec_time
        results.append(result)

    return results
