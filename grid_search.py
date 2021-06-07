import model
import numpy as np

def grid_hyperparameters(param, n=5):
    for name, (start, end) in param.items():    
        if start > end:
            t = start
            start = end
            end = t
            
        if isinstance(start, int) and isinstance(end, int):
            param[name] = np.linspace(start, end + 1, n, endpoint=False, dtype=int)
        else:
            param[name] = np.linspace(start, end, n)
            
    return param


def grid_search(param, limit=10000):
    p = dict()
    result = dict()
    count = np.zeros(len(param), dtype=np.int)
    
    while limit > 0:
        limit = limit - 1
        
        for i, k in enumerate(param.keys()):
            p[k] = param[k][count[i]]

        result[str(dict(zip(p.keys(), p.values())))] = model.run(**p)
        
        for i, k in enumerate(param.keys()):
            count[i] = count[i] + 1
            
            if count[i] < len(param[k]):
                break
                
            if i == len(param) - 1:
                return result
            
            count[i] = 0
