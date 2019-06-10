import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


class ConvBase(nn.Module):
    def __init__(self, 
                 act7=nnfunc.relu, 
                 act9=nnfunc.softmax, 
                 init_func=init.xavier_normal_):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        init_func(self.conv1.weight)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        init_func(self.conv2.weight)
        self.fc1 = nn.Linear(4*4*50, 500)
        init_func(self.fc1.weight)
        self.fc2 = nn.Linear(500, 10)
        init_func(self.fc2.weight)
        self.act7 = act7
        self.act9 = act9
    
    def forward(self, input):
        input = nnfunc.max_pool2d(nnfunc.relu(self.conv1(input)), 2, 2)
        input = nnfunc.max_pool2d(nnfunc.relu(self.conv2(input)), 2, 2)
        input = self.fc2(self.act7(self.fc1(input.view(-1, 4*4*50))))
        output = self.act9(input)
        return output

def train(model, train_loader, opt, loss_func, device, epoch, length):
    model.train()
    loss_mean = 0.0
    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        opt.zero_grad()
        # eval
        prediction = model(data)
        # compute loss and grad
        loss = loss_func(prediction, label)
        loss.backward()
        # gradient descent performed
        opt.step()
        loss_mean += loss.item()
        if i % 100 == 0 :
            print('\tTrain Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, 
                i * len(data), 
                length,
                loss.item()))
    print('\tTrain avg. loss in this epoch : {:7.5f}'.format(loss_mean / len(train_loader)))

def test(model, test_loader, loss_func, device, isValid, length):
    model.eval()
    loss_sum, acc = 0.0, 0.0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            o = model(data)
            prediction = o.argmax(dim = 1, keepdim = True)
            loss_sum += loss_func(o, label, reduction = 'sum').item()
            acc += prediction.eq(label.view_as(prediction)).sum().item()
    tot = length
    print('\t{} : avg. loss: {:7.5f}, acc: {:6.4f}\n'.format('Validation' if isValid else 'Test', loss_sum/tot, 100 * acc/tot))

def train_return(model, train_loader, opt, loss_func, device, epoch, length):
    model.train()
    loss_mean = 0.0
    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        opt.zero_grad()
        # eval
        prediction = model(data)
        # compute loss and grad
        loss = loss_func(prediction, label)
        loss.backward()
        # gradient descent performed
        opt.step()
        loss_mean += loss.item()
        if i % 100 == 0 :
            print('\tTrain Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, 
                i * len(data), 
                length,
                loss.item()))
    loss_mean /= len(train_loader)
    return loss_mean

def test_return(model, test_loader, loss_func, device, isValid, length):
    model.eval()
    loss_sum, acc = 0.0, 0.0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            o = model(data)
            prediction = o.argmax(dim = 1, keepdim = True)
            loss_sum += loss_func(o, label, reduction = 'sum').item()
            acc += prediction.eq(label.view_as(prediction)).sum().item()
    tot = length
    print('\t{} : avg. loss: {:7.5f}, acc: {:6.4f}\n'.format('Validation' if isValid else 'Test', loss_sum/tot, 100 * acc/tot))

    return loss_sum/tot

def main():
    lr = 0.01
    momentum = 0.5
    epoch = 5
    torch.manual_seed(1)
    device = torch.device("cuda")
    kwargs = {'num_workers':1, 'pin_memory':True}

    def my_normal_(weights):
        return init.normal_(weights, 0, 0.01)
    def my_uniform_(weights):
        return init.uniform_(weights, 0, 0.01)
    def my_kaiming_normal_(weights):
        return init.kaiming_normal_(weights, a=0.2)

    # validation set split
    rawset = datasets.MNIST('../data', train=True, download=True, 
                            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    rawset_len = len(rawset)
    idxs = list(range(rawset_len))
    validation_len = int(np.floor(0.1 * rawset_len))
    validation_idxs = np.random.choice(idxs, size=validation_len, replace=False)
    train_idxs = list(set(idxs) - set(validation_idxs))
    train_sampler = SubsetRandomSampler(train_idxs)
    validation_sampler = SubsetRandomSampler(validation_idxs)
    train_len = len(train_idxs)

    print('\n>>>>> Experiment 1 : Batch size <<<<<\n')
    for batch in (1, 4, 16, 32):
        print('\n>>> Batch size : {}'.format(batch))
        train_loader = torch.utils.data.DataLoader(
            rawset, sampler=train_sampler, batch_size=batch, **kwargs
        )
        validation_loader = torch.utils.data.DataLoader(
            rawset, sampler=validation_sampler, batch_size=1000, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                            transform = transforms.Compose([transforms.ToTensor(), 
                                                            transforms.Normalize((0.1307,), (0.3081,))])), 
            batch_size=1000, shuffle=True, **kwargs
        )
        model = ConvBase().to(device)
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        for epoch in range(1, 5):
            etl = train_return(model, train_loader, opt, nnfunc.cross_entropy, device, epoch, train_len)
            evl = test_return(model, validation_loader, nnfunc.cross_entropy, device, True, validation_len)
            print('\tepoch {} : train avg loss : {:7.5f}, validation avg loss {:7.5f}'.format(epoch, etl, evl))
        test(model, test_loader, nnfunc.cross_entropy, device, False, len(test_loader)*1000)

    train_loader = torch.utils.data.DataLoader(
        rawset, sampler=train_sampler, batch_size=32, **kwargs
    )
    validation_loader = torch.utils.data.DataLoader(
        rawset, sampler=validation_sampler, batch_size=1000, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                        transform = transforms.Compose([transforms.ToTensor(), 
                                                        transforms.Normalize((0.1307,), (0.3081,))])), 
        batch_size=1000, shuffle=True, **kwargs
    )

    print('\n>>>>> Learning Curve - epoch 30, relu-softmax-xavier-cross_entropy, gaussian init <<<<<\n')
    model_plot = ConvBase(init_func=my_normal_).to(device)
    opt = optim.SGD(model_plot.parameters(), lr=lr, momentum=momentum)
    train_loss_history = []
    test_loss_history = []
    for epoch in range(1, 31):
        epoch_train_loss = train_return(model_plot, train_loader, opt, nnfunc.cross_entropy, device, epoch, train_len)
        epoch_test_loss = test_return(model_plot, validation_loader, nnfunc.cross_entropy, device, True, validation_len)
        train_loss_history.append(epoch_train_loss)
        test_loss_history.append(epoch_test_loss)
    plt.plot(train_loss_history, label = 'loss_train')
    plt.plot(test_loss_history, label = 'loss_validation')
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    print('\n>>>>> Experiment 2 : activation-7 <<<<<\n')
    for act_name, actfunc in {'sigmoid':nnfunc.sigmoid, 'tanh':nnfunc.tanh, 'relu':nnfunc.relu}.items():
        print('\n>>> activation 7 function : {}'.format(act_name))
        model = ConvBase(act7=actfunc).to(device)
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        for epoch in range(1, 11):
            train(model, train_loader, opt, nnfunc.cross_entropy, device, epoch, train_len)
            test(model, validation_loader, nnfunc.cross_entropy, device, True, validation_len)
        test(model, test_loader, nnfunc.cross_entropy, device, False, len(test_loader)*1000)
    
    print('>>>>> Experiment 3 : activation-9 & loss function <<<<<\n')
    for act_name, actfunc in {'sigmoid':nnfunc.sigmoid, 'softmax':nnfunc.softmax}.items():
        print('\n>>> activation 9 function : {}'.format(act_name))
        model = ConvBase(act9=actfunc).to(device)
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
        def one_hot_l1(output, label, reduction='mean'):
            one_hot_label = nnfunc.one_hot(label, 10)
            return nnfunc.l1_loss(output, one_hot_label.float(), reduction=reduction)
        def one_hot_l2(output, label, reduction='mean'):
            one_hot_label = nnfunc.one_hot(label, 10)
            return nnfunc.mse_loss(output, one_hot_label.float(), reduction=reduction)
        
        for loss_name, loss_func in {'cross entropy': nnfunc.cross_entropy, 'l1': one_hot_l1, 'l2':one_hot_l2}.items():
            print('\tloss function : {}'.format(loss_name))
            for epoch in range(1, 11):
                train(model, train_loader, opt, loss_func, device, epoch, train_len)
                test(model, validation_loader, loss_func, device, True, validation_len)
            test(model, test_loader, loss_func, device, False, len(test_loader)*1000)
    
    print('>>>>> Experiment 4 : initialization <<<<<\n')
    for init_name, init_func in {'uniform':my_uniform_, 'gaussian':my_normal_, 'xavier':init.xavier_normal_, 'msra':my_kaiming_normal_}.items():
        print('\n>>> initialization function : {}'.format(init_name))
        model = ConvBase(init_func=init_func).to(device)
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        for epoch in range(1, 11):
            train(model, train_loader, opt, nnfunc.cross_entropy, device, epoch, train_len)
            test(model, validation_loader, nnfunc.cross_entropy, device, True, validation_len)
        test(model, test_loader, nnfunc.cross_entropy, device, False, len(test_loader)*1000)

if __name__ == '__main__':
    main()
        




