import numpy as np
import torch
from  torch.utils.data import Dataset,DataLoader
import math
import torchvision
#1.transform data into tensor
'''
data = torchvision.dataset.MNIST(root='./data',transform=torchvision.transform.ToTensor())
'''


class WineDataset(Dataset):

    def __init__(self,transform=None):
        xy = np.loadtxt('wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = xy[:,1:]
        self.y = xy[:,[0]]#n_samples,1
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample =  self.x[index],self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_samples

#2. custom transform   
class ToTensor:

    def __call__(self,sample):
        inputs,target = sample
        return torch.from_numpy(inputs),torch.from_numpy(target)
    

#3. multiple transform

class MulTransform:

    def __init__(self,factor):
        self.factor = factor
    
    def __call__(self,sample):
        inputs,target = sample
        inputs *= self.factor
        return inputs,target

composed = torchvision.transforms.Compose([ToTensor(),MulTransform(2)])


# dataset = WineDataset(transform=ToTensor())
dataset = WineDataset(transform=composed)
first_data = dataset[0]
inputs,target =first_data
print("inputs : ",inputs)

