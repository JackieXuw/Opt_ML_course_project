import numpy as np
import model
def grid_hyperparam(param,n=5):
    for name, (start, end) in param.items():
        if start> end:
            t=start
            start=end
            end=t
        if start==int(start) and end==int(end):
            step=int(np.abs(end-start)/float(n))
            if step==0:
                step=1
            param[name]=np.arange(start,end,step)
        else:
            param[name] = np.arange(start, end, np.abs(end - start) / float(n))
    return param

def grid_search(param,limit=10000):
    p = dict()
    result = dict()
    count = np.zeros(len(param), dtype=np.int)
    while limit>0:
        limit=limit-1
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


param=dict()
param['lr']=[0.01,0.1]
param['nb_epochs']=[100,500]
param=grid_hyperparam(param,2)
print(grid_search(param))