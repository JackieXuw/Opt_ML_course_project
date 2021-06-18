import math
import numpy as np

from model import *
from torch import optim
from results import *

# It initializes the hyperparameters.
# The input parameters_range is a dictionary with the parameters' names as keys and their types and range as values.
# It returns a new dictionary with the parameters' names as keys and the set of possible values as values.
# Each parameter has equidistant possible values.
def grid_hyperparameters(parameters_range):
    parameters = dict()

    for name, (start, end, grid_size, space) in parameters_range.items():
        if start > end:
            t = start
            start = end
            end = t

        if space[:17] == 'discrete_linspace':
            parameters[name] = np.linspace(start, end, grid_size, dtype=int)
        elif space[:8] == 'linspace':
            parameters[name] = np.linspace(start, end, grid_size)
        elif space[:17] == 'discrete_logspace':
            base = int(space[18:])
            start = math.log(start, base)
            end = math.log(end, base)
            parameters[name] = np.logspace(start, end, grid_size, base=base, dtype=int)
        elif space[:8] == 'logspace':
            base = int(space[9:])
            start = math.log(start, base)
            end = math.log(end, base)
            parameters[name] = np.logspace(start, end, grid_size, base=base)
        elif space == 'fixed' and grid_size == 1 and start == end:
            parameters[name] = [start]
        else:
            raise ValueError

    return parameters


# It takes as input the output dictionary of grid_hyperparameters().
# In each iteration, it computes the errors and the execution time of the values' combinations,
# and it stores them into a new dictionary.
# Once the operations are completed, or the limit is exceeded, it returns the dictionary.
def grid_search(parameters, limit=1000, offline=False):
    p = dict()
    results = []
    count = np.zeros(len(parameters), dtype=np.int)
    t=time.time()

    while limit > 0:
        limit = limit - 1

        for i, k in enumerate(parameters.keys()):
            p[k] = parameters[k][count[i]]
        if offline:
            train_error, test_error, exec_time = get_results(lr=p['lr'], momentum=p['momentum'],num_hidden=p['num_hidden'], num_layers=p['num_layers'], mini_batch_size=p['mini_batch_size'],num_epochs=p['num_epochs'])
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
            result['run_time'] = time.time() - t +exec_time
        else:
            result['run_time'] = time.time() - t
        t=time.time()
        results.append(result)

        for i, k in enumerate(parameters.keys()):
            count[i] = count[i] + 1

            if count[i] < len(parameters[k]):
                break

            if i == len(parameters) - 1:
                return results

            count[i] = 0
    return results

