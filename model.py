import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets
import time

PARAMETER_NAMES = {'lr', 'momentum', 'mini_batch_size', 'num_epochs', 'num_hidden', 'num_layers'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # it would run on gpu if available


# Load mnist data
# The dataset consists of pictures representing the numbers 0-9, and the return values are normalized.
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


# Deep learning network
# It takes as input parameters the number of hidden units, the number of hidden layers and the dropout value.
# It is composed by two convolutional layers and other linear layers
class Net(nn.Module):
    def __init__(self, num_hidden=100, num_layers=2, dropout=0.3):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc_i = nn.Linear(256, num_hidden)
        self.fcs = []
        for i in range(num_layers - 2):
            f = nn.Linear(num_hidden, num_hidden).to(device)
            self.fcs.append(f)
        self.fc_o = nn.Linear(num_hidden, 10)

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


# Model training process with MSELoss criterion
def train_model(model, optimizer, train_input, train_target, mini_batch_size=100, num_epochs=100):
    criterion = nn.MSELoss()
    size = int(train_input.size(0) / mini_batch_size) * mini_batch_size

    for e in range(num_epochs):
        acc_loss = 0
        for b in range(0, size, mini_batch_size):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()


# Error computation
def compute_nb_errors(model, input, target):
    nb_errors = 0

    output = model(input)
    _, predicted_classes = output.max(1)
    for k in range(input.size(0)):
        if target[k, predicted_classes[k]] <= 0:
            nb_errors = nb_errors + 1

    return nb_errors


# It converts the targets into one-hot format
def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)

    return tmp


train_i, train_t, test_i, test_t = load_data()


# It takes as input the model and the optimizer.
# It trains the model, and it returns the train errors, the test errors and the execution time.
def run(model, optimizer, mini_batch_size=100, num_epochs=300):
    train_input, train_target = train_i.to(device), train_t.to(device)
    test_input, test_target = test_i.to(device), test_t.to(device)
    model.to(device)
    start = time.time()
    model.train()
    train_model(model, optimizer, train_input, train_target, mini_batch_size=mini_batch_size, num_epochs=num_epochs)
    exec_time = (time.time() - start)
    nb_train_errors = compute_nb_errors(model.eval(), train_input, train_target)
    nb_test_errors = compute_nb_errors(model.eval(), test_input, test_target)
    return nb_train_errors / 10, nb_test_errors / 10, exec_time

