from model import *
from torch import optim
import matplotlib.pyplot as plt

        
def visualize_experiment(results):

    train_data = list(map(lambda x: x['train_error'], results))
    test_data = list(map(lambda x: x['test_error'], results))
    time_data = list(map(lambda x: x['exec_time'], results))
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    X = range(1, len(train_data) + 1)

    ax1.plot(X, train_data, label='Train', color='orange')
    ax1.plot(X, test_data, label='Test', color='green')
    ax2.plot(X, time_data, label='Time', color='red')
    
    plt.xlabel('Trial number')
    ax1.set_ylabel('Error [%]')
    ax2.set_ylabel('Time [ms]')
    plt.xticks(X)
    
    def color_y_axis(ax, color):
        for t in ax.get_yticklabels():
            t.set_color(color)

    color_y_axis(ax2, 'r')
    
    ax1.legend()
    ax2.legend()
    plt.show()