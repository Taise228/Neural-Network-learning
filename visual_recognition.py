from random import shuffle
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn as nn
import torch.nn.functional as F

# prepare a dataset
ds = datasets.FashionMNIST(root="data",
                           train=True, 
                           download=True, 
                           transform = ToTensor(),
                           target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float32).scatter_(0, torch.tensor(y), value=1)))

# divide dataset into the train section and the test section
n_train = int(len(ds) *  0.8)
n_test = len(ds) - n_train
train, test = torch.utils.data.random_split(ds, [n_train, n_test])

#DataLoader ... collect several data together in one batch to prevent local minimum
train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=10)