import sys

import torch
import torchvision.utils

from models import lenet as net
from Utils.data import load_data_fashion_mnist
from torch import nn
from Utils.train import Accumulator, accuracy, evaluate_accuracy_gpu, Timer, try_gpu
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

batch_size = 64
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
examples = iter(train_iter)

example_data, example_targets = next(examples)
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
writer.add_graph(net, example_data)


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)  # loss
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)

        writer.add_scalar('Loss', train_l, epoch)
        writer.add_scalar('Train Accuracy', train_acc, epoch)
        writer.add_scalar('Test Accuracy', test_acc, epoch)


lr, num_epochs = 0.9, 20
train(net, train_iter, test_iter, num_epochs, lr, try_gpu())
torch.save(net.state_dict(), 'lenet.pt')

writer.close()
