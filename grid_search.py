import numpy as np

from model import *
from torch import optim


def grid_hyperparameters(parameters_range, grid_size=5):
    parameters = dict()
    
    for name, (start, end) in parameters_range.items():    
        if start > end:
            t = start
            start = end
            end = t
            
        if isinstance(start, int) and isinstance(end, int):
            parameters[name] = np.linspace(start, end + 1, grid_size, endpoint=False, dtype=int)
        else:
            parameters[name] = np.linspace(start, end, grid_size)
            
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
