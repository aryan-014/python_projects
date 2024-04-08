import torch
import numpy as np
import torch.nn as nn

#1. softmax
#with numpy
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

arr = np.array([2,1,0])
# print(softmax(arr))

#with pytorch

x = torch.tensor([2.0,1.0,0.0])
out = torch.softmax(x,dim=0)
print(out)

#2. cross entropy loss

#with numpy
def cross_entropy(actual,predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

actual = np.array([1,0,0])

#y_pred has possibilities

y_pred_good = np.array([0.8,0.1,0.1])
y_pred_bad = np.array([0.1,0.2,0.7])

l1 = cross_entropy(actual,y_pred_good)
l2 = cross_entropy(actual,y_pred_bad)

# print(f'loss1 entropy : {l1:.3f}')
# print(f'loss2 entropy : {l2:.3f}')

#with pytorch

# 1. num of sample = 1,num of class = 3 ,1*3
# actual = torch.tensor([2])#not hot encoded

# pred_good = torch.tensor([[2.0,4.2,1.1]])#only scores ,not softmax
# pred_bad = torch.tensor([[0.1,1.2,3.2]])#only scores ,not softmax

# 2. num of sample = 3,num of class = 3 ,3*3
actual = torch.tensor([2,0,1])
pred_good = torch.tensor([[2.0,1.1,4.2],[3.0,1.1,2.2],[2.0,3.1,1.2]])#only scores ,not softmax
pred_bad = torch.tensor([[3.0,1.1,2.2],[2.0,3.1,1.2],[2.0,1.1,4.2]])#only scores ,not softmax

loss = nn.CrossEntropyLoss()

loss1 = loss(pred_good,actual)
loss2 = loss(pred_bad,actual)

print(f'loss1 entropy : {loss1.item():.3f}')
print(f'loss2 entropy : {loss2.item():.3f}')

#neural network

class MulitClassNN(nn.Module):

    def __init__(self,input_size,hidden_size,num_of_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,num_of_classes)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #no softwar in end (bcoz we use cross entropy)
        return out
    
model = MulitClassNN(input_size=28*28,hidden_size=5,num_of_classes=3)
criterion = nn.CrossEntropyLoss()#applied softmax

class BinarClassNN(nn.Module):

    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #sigmod in end (we use BCEloss)
        y_pred = torch.sigmoid(out)
        return y_pred
    
model = BinarClassNN(input_size=28*28,hidden_size=5)
criterion = nn.BCELoss()#applied softmax
