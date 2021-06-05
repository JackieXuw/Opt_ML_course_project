import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
import datetime

class Net(nn.Module):
    def __init__(self,nb_hidden = 100, nb_layers=2, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc_i = nn.Linear(256, nb_hidden)
        self.fcs=[]
        for i in range(nb_layers-2):
            self.fcs.append(nn.Linear(nb_hidden, nb_hidden))
        self.fc_o = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.dropout(x)
        x = F.relu(self.fc_i(x.view(-1, 256)))
        x = self.dropout(x)
        for fc in self.fcs:
            x = fc(x)
            x = self.dropout(x)
        x = self.fc_o(x)
        return x


def train_model(model, train_input, train_target, lr=1e-1, mini_batch_size=100, nb_epochs=100):
    criterion = nn.MSELoss()

    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad



def compute_nb_errors(model, input, target, mini_batch_size=100):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def load_data():
    mnist_train_set = datasets.MNIST('./data/mnist/', train=True, download=True)
    mnist_test_set = datasets.MNIST('./data/mnist/', train=False, download=True)

    train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
    train_target = mnist_train_set.targets
    test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
    test_target = mnist_test_set.targets

    train_input = train_input.narrow(0, 0, 1000)
    train_target = train_target.narrow(0, 0, 1000)
    test_input = test_input.narrow(0, 0, 1000)
    test_target = test_target.narrow(0, 0, 1000)

    train_target = convert_to_one_hot_labels(train_input, train_target)
    test_target = convert_to_one_hot_labels(test_input, test_target)

    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    return train_input, train_target, test_input, test_target


def run(lr=1e-1,mini_batch_size=100,nb_epochs=100,nb_hidden=100,nb_layers=2,dropout=0.):
    train_input, train_target, test_input, test_target = load_data()
    model = Net(nb_hidden=nb_hidden,nb_layers=nb_layers, dropout=dropout)
    start = datetime.datetime.now()
    model.train()
    train_model(model, train_input, train_target,lr=lr,mini_batch_size=mini_batch_size,nb_epochs=nb_epochs)
    nb_train_errors = compute_nb_errors(model.eval(), train_input, train_target)
    nb_test_errors= compute_nb_errors(model.eval(), test_input, test_target)
    exec_time = (datetime.datetime.now() - start).microseconds

    return nb_train_errors/10,nb_test_errors/10,exec_time


#for i in range(5):
   #run()
