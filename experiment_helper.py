from model import *
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import time


# It takes as input the dictionary with the combination of parameters as keys, and the errors and the execution time as values.
# It displays the results by plotting the values.
def visualize_experiment(results, name=None):
    if name is not None:
        save(results, name)
    train_data = list(map(lambda x: x['train_error'], results))
    test_data = list(map(lambda x: x['test_error'], results))
    time_data = list(map(lambda x: x['exec_time'], results))

    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax2 = ax1.twinx()
    X = range(1, len(train_data) + 1)

    ax1.plot(X, train_data, label='Train', color='orange')
    ax1.plot(X, test_data, label='Test', color='green')
    ax2.plot(X, time_data, label='Time', color='red')

    plt.xlabel('Trial number')
    ax1.set_ylabel('Error [%]')
    ax2.set_ylabel('Time [s]')
    plt.xticks(X)

    def color_y_axis(ax, color):
        for t in ax.get_yticklabels():
            t.set_color(color)

    color_y_axis(ax2, 'r')

    ax1.legend()
    ax2.legend()
    plt.show()


def visualize_run_time(results):
    '''
    Plot the test error in results against the running time used to attain it.
    Data points are plotted every 60 seconds.
    '''
    max_t = 0.0
    for r in results:
        max_t += r['run_time']

    max_t = int(max_t / 60)
    x = range(0, max_t + 1)
    y = np.zeros(len(x))
    min = 100
    index = 0
    time = 0.0
    for r in results:
        time += r['run_time']
        if int(time / 60) > index:
            y[index] = min
            index += 1
        if r['test_error'] < min:
            min = r['test_error']

    for i in range(index, len(y)):
        y[i] = min

    plt.plot(x, y, color='orange')
    plt.show()
    
    return x, y


def keep_best_results(results):
    decreasing_results = []
    min_train_error = results[0]['train_error']
    min_test_error = results[0]['test_error']
    min_exec_time = results[0]['exec_time']
    max_t = 0.0
    for r in results:
        max_t += r['run_time']

    max_t = int(max_t / 20)
    time = 0.0
    index = 0

    for result in results:
        time += result['run_time']
        if int(time / 20) > index:
            decreasing_result = dict()
            decreasing_result['train_error'] = min_train_error
            decreasing_result['test_error'] = min_test_error
            decreasing_result['exec_time'] = min_exec_time
            decreasing_results.append(decreasing_result)
            index += 1
        min_exec_time = result['exec_time'] if result['exec_time'] < min_exec_time else min_exec_time
        min_test_error = result['test_error'] if result['test_error'] < min_test_error else min_test_error
        min_train_error = result['train_error'] if result['train_error'] < min_train_error else min_train_error

    return decreasing_results


def get_plotting_data(list_of_results):
    train_errors = []
    test_errors = []
    exec_times = []
    for trials in zip(*[keep_best_results(a) for a in list_of_results]):
        train_errors.append([trial['train_error'] for trial in trials])
        test_errors.append([trial['test_error'] for trial in trials])
        exec_times.append([trial['exec_time'] for trial in trials])

    decreasing_mean_train_errors = [np.mean(train_error) for train_error in train_errors]
    quantile_train_errors = np.array(
        [(np.quantile(train_error, 0.25), np.quantile(train_error, 0.75)) for train_error in train_errors]).transpose()
    quantile_train_errors = np.abs(quantile_train_errors - decreasing_mean_train_errors)

    decreasing_mean_test_errors = [np.mean(test_error) for test_error in test_errors]
    quantile_test_errors = np.array(
        [(np.quantile(test_error, 0.25), np.quantile(test_error, 0.75)) for test_error in test_errors]).transpose()
    quantile_test_errors = np.abs(quantile_test_errors - decreasing_mean_test_errors)

    decreasing_mean_exec_times = [np.mean(exec_time) for exec_time in exec_times]
    quantile_exec_times = np.array(
        [(np.quantile(exec_time, 0.25), np.quantile(exec_time, 0.75)) for exec_time in exec_times]).transpose()
    quantile_exec_times = np.abs(quantile_exec_times - decreasing_mean_exec_times)

    return decreasing_mean_train_errors, quantile_train_errors, decreasing_mean_test_errors, quantile_test_errors, decreasing_mean_exec_times, quantile_exec_times


def plot_all_results(random_search_results, grid_search_results, bayesian_optimiser_results):
    m1_r, q1_r, m2_r, q2_r, m3_r, q3_r = get_plotting_data(random_search_results)
    m1_g, _, m2_g, _, m3_g, _ = get_plotting_data(grid_search_results)
    m1_b, q1_b, m2_b, q2_b, m3_b, q3_b = get_plotting_data(bayesian_optimiser_results)

    X = range(1, max(len(m1_r), len(m1_g), len(m1_b)) + 1)
    x=np.array(X)*20
    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(311)
    ax1.set_xlabel('Run time[s]')
    ax1.set_ylabel('Train error [%][dB]')
    ax1.set_xticks(x)
    ax1.set_yscale('log')
    plt.errorbar(x[:len(m1_r) ], (np.array(m1_r)+0.0001), yerr=(np.array(q1_r)+0.00001), c='m', label='Random search')
    plt.plot(x[:len(m1_g) ], (np.array(m1_g)+0.0001), c='orange', label='Grid search')
    plt.errorbar(x[:len(m1_b) ], (np.array(m1_b)+0.0001), yerr=(np.array(q1_b)+0.00001), c='green', label='Bayesian optimisation')

    ax1.legend()

    ax2 = plt.subplot(312)
    ax2.set_xlabel('Run time[s]')
    ax2.set_ylabel('Test error [%][dB]')
    ax2.set_xticks(x)
    ax2.set_yscale('log')
    plt.errorbar(x[:len(m2_r)], (np.array(m2_r)+0.0001), yerr=(np.array(q2_r)+0.00001), c='m', label='Random search')
    plt.plot(x[:len(m2_g)], (np.array(m2_g)+0.0001), c='orange', label='Grid search')
    plt.errorbar(x[:len(m2_b)], (np.array(m2_b)+0.0001), yerr=(np.array(q2_b)+0.00001), c='green', label='Bayesian optimisation')
    ax2.legend()

    ax3 = plt.subplot(313)
    ax3.set_xlabel('Run time[s]')
    ax3.set_ylabel('Exec time [s]')
    ax3.set_xticks(x)
    plt.errorbar(x[:len(m3_r)], m3_r, yerr=q3_r, c='m', label='Random search')
    plt.plot(x[:len(m3_g)], m3_g, c='orange', label='Grid search')
    plt.errorbar(x[:len(m3_b)], m3_b, yerr=q3_b, c='green', label='Bayesian optimisation')
    ax3.legend()
    t = time.strftime("%Y%m%d%H%M%S", time.localtime())
    plt.savefig('figures/fig-' + t)


def compare(a, b):
    return a[-4] + a[-3] + np.log10(a[-2]) < b[-4] + b[-3] + np.log10(b[-2])


def choose_best_hyperparam(results):
    d = None
    e = None
    r = None
    for v in results:
        if d is None:
            d = v.keys()
            e = v.values()
            r = v
        else:
            a = list(v.values())
            b = list(e)
            if compare(a, b):
                d = v.keys()
                e = v.values()
                r = v
    d = list(d)
    e = list(e)
    return dict(zip(d[-1:-4], e[-1:-4])), r


def save(results,name=' '):
    a = time.strftime("%Y%m%d%H%M%S.txt", time.localtime())
    f = open('results/' + name + a, mode="w")
    f.write(str(results))
    f.close()


