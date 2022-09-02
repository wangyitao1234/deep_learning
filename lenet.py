import torch
import time

import torch.utils.data
import torchvision
from torchvision import datasets,transforms
from torch import nn,optim

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc =nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )

    def forward(self, img):
        feature =self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1),)
        return output

batch_size =256
mnist_train =torchvision.datasets.FashionMNIST(root='E:/pytorch_ws/fashion_mnist',train=True,download=True,transform=torchvision.transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='E:/pytorch_ws/fashion_mnist',train=True,download=True,transform=torchvision.transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size,shuffle = True)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size,shuffle = True)



def evaluate_accuracy(data_iter,net,device=None):
    if device is None and isinstance(net,torch.nn.Module):
        device =list(net.parameters())[0].device
    acc_sum,n=0.0,0
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(net,torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X,is_training=False).argmax(dim=1) == y).float().sum()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n +=y.shape[0]
    return acc_sum/n

def train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    net =net.to(device)
    print("training on",device)
    loss =torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,batch_count,start =0.0,0.0,0,0,time.time()
        for X,y in train_iter:
            X =X.to(device)
            y =y.to(device)
            y_hat =net(X)
            l =loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum +=l.cpu().item()
            train_acc_sum +=(y_hat.argmax(dim=1) == y).sum().cpu().item()
            n +=y.shape[0]
            batch_count +=1
        test_acc =evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

net =LeNet()
lr,num_epochs =0.001,5
optimizer =torch.optim.Adam(net.parameters(),lr=lr)
train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)




