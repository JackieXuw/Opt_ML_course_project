from model import *
from torch import optim
import matplotlib.pyplot as plt

def run_experiment(parameters):
    train_data = []
    test_data = []
    time_data = []

    num_trials = len(parameters['num_hidden'])
    for i in range(num_trials):
        model = Net(num_hidden=parameters['num_hidden'][i], num_layers=parameters['num_layers'][i])
        sgd = optim.SGD(model.parameters(), lr=parameters['lr'][i], momentum=parameters['momentum'][i])
        train_error, test_error, time = run(model, sgd, 
                                            mini_batch_size=parameters['mini_batch_size'][i], 
                                            num_epochs=parameters['num_epochs'][i])
        train_data.append(train_error)
        test_data.append(test_error)
        time_data.append(time)

    return train_data, test_data, time_data
        
def visualize_experiment(train_data, test_data, time_data):

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