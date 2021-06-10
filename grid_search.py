import math
import numpy as np

from model import *
from torch import optim


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


def grid_search(parameters, limit=10000):
    p = dict()
    results = []
    count = np.zeros(len(parameters), dtype=np.int)
    
    while limit > 0:
        limit = limit - 1
        
        for i, k in enumerate(parameters.keys()):
            p[k] = parameters[k][count[i]]
            
        model = Net(num_hidden=p['num_hidden'], num_layers=p['num_layers'])
        sgd = optim.SGD(model.parameters(), lr=p['lr'], momentum=p['momentum'])
        train_error, test_error, exec_time = run(model, sgd, mini_batch_size=p['mini_batch_size'], num_epochs=p['num_epochs'])
        result = dict(zip(p.keys(), p.values()))
        result['train_error'] = train_error
        result['test_error'] = test_error
        result['exec_time'] = exec_time
        results.append(result)

        
        for i, k in enumerate(parameters.keys()):
            count[i] = count[i] + 1
            
            if count[i] < len(parameters[k]):
                break
                
            if i == len(parameters) - 1:
                return results
            
            count[i] = 0
