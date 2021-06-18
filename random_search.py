import math
import numpy as np
from model import *
from torch import optim
from results import *

# It initializes the hyperparameters.
# The input parameters_range is a dictionary with the parameters' names as keys, and their types and range as values.
# It returns a new dictionary with the parameters' names as keys and the set of possible values as values.
# The possible values are picked randomly in the range.
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

# It creates n discrete uniform random variables into the range [low,high].
def generate_discrete_uniform(low, high, n):
    return np.random.random_integers(low, high, (n,))

# It creates n uniform random variables into the range [low,high].
def generate_uniform(low, high, n):
    return np.random.uniform(low, high, (n,))

# It creates n uniform random variables into the range [log(low),log(high)], with a defined base for the logarithm.
# It returns base^(the created value).
def generate_loguniform(low, high, n, base):
    return np.power(base, generate_uniform(math.log(low, base), math.log(high, base), n))

# It creates n discrete uniform random variables into the range [log(low),log(high)], with a defined base for the logarithm.
# It returns base^(the created value).
def generate_discrete_loguniform(low, high, n, base):
    return np.power(base, generate_discrete_uniform(np.floor(math.log(low, base)), np.floor(math.log(high, base)), n).astype(float)).astype(int)

# It takes as input the output dictionary of random_hyperparameters().
# In each iteration, it computes the errors and the execution time of the values' combinations,
# and it stores them into a new dictionary.
# Once the operations are completed, or the limit is exceeded, it returns the dictionary.
def random_search(parameters,offline=False):
    p = dict()
    #results = dict()
    results = list()
    t=time.time()
    num_trials = len(parameters['num_hidden'])
    for i in range(num_trials):
        for k in parameters.keys():
            p[k] = parameters[k][i]

        if offline:
            train_error, test_error, exec_time = get_results(lr=p['lr'], momentum=p['momentum'], num_hidden=p['num_hidden'],
                                                         num_layers=p['num_layers'],
                                                         mini_batch_size=p['mini_batch_size'],
                                                         num_epochs=p['num_epochs'])
        else:
            model = Net(num_hidden=p['num_hidden'], num_layers=p['num_layers'])
            sgd = optim.SGD(model.parameters(), lr=p['lr'], momentum=p['momentum'])
            train_error, test_error, exec_time = run(model, sgd, mini_batch_size=p['mini_batch_size'],
                                                     num_epochs=p['num_epochs'])
        result = dict(zip(p.keys(), p.values()))
        result['train_error'] = train_error
        result['test_error'] = test_error
        result['exec_time'] = exec_time
        if offline:
            result['run_time'] = time.time() - t + exec_time
        else:
            result['run_time'] = time.time() - t
        t = time.time()
        results.append(result)

    return results

