from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import time
from my_function import *
import os
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __getitem__(self, index):#返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target
    def __len__(self):
        return len(self.images)

# BiRNN Model (Many-to-One)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()) # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda())
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])

        return out

if __name__ == '__main__':
    # Hyper Parameters
    t0 = time.time()
    sequence_length = 169
    input_size = 16
    hidden_size = 120
    num_layers = 2
    num_classes = 6
    batch_size = 16
    num_epochs = 1
    learning_rate = 0.001
    path = '/home/leyuan/文档/data/input-train'
    train_data = os.listdir(path)
    train_set, test_set = train_test_split(train_data, random_state=1, train_size=0.7)
    # X, Y, Y0 = loaddata(path, train_set, y0_train=True)
    # np.save('X-tr.npy', X)
    # np.save('Y-tr.npy', Y)
    # np.save('Y0-tr.npy', Y0)
    X = np.load('./bi-rnn/X-tr.npy')
    Y = np.load('./bi-rnn/Y-tr.npy')
    dataset = MyDataset(X, Y)
    # XX, YY, Y0 = loaddata(path, train_set, y0_train=True)
    # np.save('X-te.npy', X)
    # np.save('Y-te.npy', Y)
    # np.save('Y0-te.npy', Y0)
    XX = np.load('./bi-rnn/X-te.npy')
    YY = np.load('./bi-rnn/Y-te.npy')
    testset = MyDataset(XX, YY)

    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, **kwargs)
    rnn = BiRNN(input_size, hidden_size, num_layers, num_classes)
    rnn.cuda()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train the Model
    t1 = time.time()
    print('train loader', len(train_loader), '    加载时间：', t1 - t0)
    correct_train = 0
    num = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.float()
            labels = labels.float()
            images = Variable(images.view(-1, sequence_length, input_size).cuda())
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            labels = labels.type(torch.LongTensor).cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels.data).sum()
            num += len(predicted)
            if (i + 1) % 5000 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Test Accuracy: %0.4f'
                      % (epoch + 1, num_epochs, i + 1, len(dataset) // batch_size, loss.data[0], correct_train/num),
                      '    目前训练时间：', int(time.time() - t1), '    Process: %d'%batch_size)

    t2 = time.time()
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.float()
        labels = labels.float()
        images = Variable(images.view(-1, sequence_length, input_size)).cuda()
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        labels = labels.type(torch.LongTensor).cuda()
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on train data: %0.4f' % (correct / total), '    测试总时间：', int(time.time() - t2), '    Process: %d'%batch_size)

    # Save the Model
    torch.save(rnn.state_dict(), 'rnn-%s.pkl'%batch_size)
