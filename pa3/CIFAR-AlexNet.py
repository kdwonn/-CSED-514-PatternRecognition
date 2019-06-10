import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
import torch.optim as optim
import torch.nn.init as init
import torchvision.models as torchmodel 
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def test(model, test_loader, device, length):
    model.eval()
    acc = 0.0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            o = model(data)
            prediction = o.argmax(dim = 1, keepdim = True)
            acc += prediction.eq(label.view_as(prediction)).sum().item()
    tot = length
    print('\tTest :  acc: {:6.4f}\n'.format(loss_sum/tot, 100 * acc/tot))

kwargs = {'num_classes':10}
alexNet = torchmodel.alexnet(pretrained=True, **kwargs)
torch.manual_seed(1)
device = torch.device("cuda")
kwargs = {'num_workers':1, 'pin_memory':True}
test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False,
                                transform = transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                ])
                            ), 
            batch_size=1000, shuffle=True, **kwargs
        )
test(alexNet, test_loader, device, 1000 * len(test_loader))