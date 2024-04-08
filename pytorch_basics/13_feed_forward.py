import torch
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#gpu support
device = torch.device("cpu" if torch.cuda.is_available() else "gpu")
#Hyper Parameters
n_epochs = 2
input_size = 784#28x28
hidden_size = 100
num_of_classes = 10
batch_size = 100
learning_rate = 0.001
#MNIST
train_data = torchvision.datasets.MNIST(root='./data',transform=transforms.ToTensor(),train=True,download=True)
test_data = torchvision.datasets.MNIST(root='./data',transform=transforms.ToTensor(),train=False)

train_loader = torch.utils.data.DataLoader(dataset=train_data,shuffle=True,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_data,shuffle=False,batch_size=batch_size)

#Neural Netwoek

class NeuralNet(nn.Module):

    def __init__(self,input_size,hidden_size,num_of_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_of_classes)

    def forward(self,input):
        out = self.l1(input)
        out = self.relu(out)
        out = self.l2(out)
        return out
    

model = NeuralNet(input_size,hidden_size,num_of_classes)
#loss and optimzer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

#show samples
# sample = iter(train_loader)
# images,labels = next(sample)
# print(f'image shape : {images.shape}')

# for i in range(6):
#     plt.subplot(2,3,i+1)#(row,col,index)
#     plt.imshow(images[i][0],cmap='gray')
    
# plt.show()

#training loop

for epoch in range(n_epochs):
    
    for i,(images,labels) in enumerate(train_loader):
        #image shape 100,1,28,28 [nums,color_channel,h,w]
        #input shape - 100,784(28*28)
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images).to(device)

        #loss
        loss = criterion(outputs,labels)

        loss.backward()
        #update weights
        optimizer.step()
        #zero grads for next iteration
        optimizer.zero_grad()

        if (i+1)%100 == 0:#every complete training loop for 100 times
            print(f'epoch : {epoch+1}/{n_epochs} , steps : {i+1}/{len(train_loader)}, loss : {loss.item():.4f}')


#find accuracy

with torch.no_grad():#don't include grad calc

    n_samples = 0
    n_correct = 0

    for i,(images,labels) in enumerate(test_loader):
        #image shape 100,1,28,28 [nums,color_channel,h,w]
        #input shape - 100,784(28*28)
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images).to(device)

        _,predictions = torch.max(outputs,1)#1 dimension
        n_correct += (predictions == labels).sum().item()
        n_samples += labels.shape[0]


    acc =   100 * n_correct/n_samples
    print(f'Accuracy : {acc:.3f}')


for i in range(6):
    sample = iter(test_loader)
    images,labels = next(sample)
    images = images[i][0]
    img = images
    images = images.reshape(-1,28*28).to(device)
    outputs = model(images)
    _,predictions = torch.max(outputs,1)
    print("predictions",predictions.item())
    text = predictions.item()
    plt.subplot(2,3,i+1)#(row,col,index)
    plt.imshow(img,cmap='gray')
    
plt.show()
