import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn as nn
import torch.nn.functional as F

#device -> gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# prepare a dataset
train_data = datasets.FashionMNIST(root="data",
                           train=True, 
                           download=True, 
                           transform = ToTensor(),
                           target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float32).scatter_(0, torch.tensor(y), value=1)))
test_data = datasets.FashionMNIST(root="data",
                           train=False, 
                           download=True, 
                           transform = ToTensor(),
                           target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float32).scatter_(0, torch.tensor(y), value=1)))
# contents of dataset = tuple(tensor(input), target)
# transform ... transform bits sequence of the image into tensor for calculating
# target_transform ... transform target value computable in this neural system (in this case, make it into one-hot signal of tensor(float32))

# divide dataset into the train section and the test section
"""n_train = int(len(ds) *  0.8)
n_test = len(ds) - n_train
train, test = torch.utils.data.random_split(ds, [n_train, n_test])"""

#DataLoader ... collect several data together in one batch to prevent local minimum
batch_size = 10
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size)

#define class of this Neural Network
class neural_network(nn.Module):
    def __init__(self) -> None:
        super(neural_network, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_mish_stack = nn.Sequential(
            nn.Linear(28*28, 512),   #28*28 = input value num
            nn.Mish(),
            nn.Linear(512, 512),
            nn.Mish(),
            nn.Linear(512, 10)   #10 = label num
        )
    
    def forward(self, x):
        x = self.flatten(x)   # turn data into 1 dimension except for the first dimension which means batch_size
        x = self.linear_mish_stack(x)
        return x

# target function
criterion = F.cross_entropy
# includes softmax

net = neural_network().to(device=device)

#select optimizer, that is, how to update parameters
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# now use SGD as an optimizer, and learning rate (lr) = 0.1

# test
'''max_epoch = 1
acc = 0
for epoch in range(max_epoch):
    for batch in train_loader:
        x, t = batch
        x = x.to(device=device)
        t = t.to(device=device)
        optimizer.zero_grad()   # initialize slopes of parametars to 0
        y = net(x)
        loss = criterion(y, t)
        print("loss: {}".format(loss.item()))
        y_label = torch.argmax(y, dim=1)   # the max target num should be a category this data belongs to
        t_label = torch.argmax(t, dim=1)
        acc += torch.sum(y_label == t_label).item()   # accurate label
        loss.backward()
        optimizer.step()

print(acc/(len(train_loader) * batch_size))'''

def train_loop(dataloader, model, criterion, optimizer, batch_size):
    max_epoch = 1
    acc = 0
    for epoch in range(max_epoch):
        for batch in dataloader:
            x, t = batch
            x = x.to(device=device)
            t = t.to(device=device)
            optimizer.zero_grad()   # initialize slopes of parametars to 0
            y = model(x)
            loss = criterion(y, t)
            #print("loss: {}".format(loss.item()))
            y_label = torch.argmax(y, dim=1)   # the max target num should be a category this data belongs to
            t_label = torch.argmax(t, dim=1)
            acc += torch.sum(y_label == t_label).item()   # num of accurate label
            loss.backward()
            optimizer.step()
    print("Accuracy: {}".format(acc/(len(dataloader)*batch_size)))

def test_loop(dataloader, model, criterion):
    size = len(dataloader.dataset)
    loss = 0
    acc = 0
    with torch.no_grad():
        for batch in dataloader:
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            y = model(x)
            y_label = torch.argmax(y, dim=1)   # the max target num should be a category this data belongs to
            t_label = torch.argmax(t, dim=1)
            loss += criterion(y, t)
            acc += torch.sum(y_label == t_label).item()
    loss /= size
    acc /= size
    print(f"Accuracy: {(100*acc):>0.2f}%, Avg loss: {loss:>8f} \n")


for epoch in range(5):
    print(f"Epoch {epoch+1} --------------------")
    train_loop(train_loader, net, criterion, optimizer, batch_size)
    test_loop(train_loader, net, criterion)