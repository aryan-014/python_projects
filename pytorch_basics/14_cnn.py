import torch
import torch.nn as nn
import torch.utils
import torchvision
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
#gpu support
device = torch.device("cpu" if torch.cuda.is_available() else "gpu")

#Hyper Parameters
num_epochs = 3
batch_size = 4
leaning_rate = 0.01

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./cipher_data', train=True,
                                        download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./cipher_data', train=False,
                                        transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#neural network
'''
conv1 = nn.Conv2d(3,6,5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(6,16,5)
linear1 = nn.Linear(16*10*10,120)
sample = iter(test_loader)
images,labels = next(sample)

print(f"Input Shape : {images.shape} ")
#[4, 3, 32, 32]
x = conv1(images)
print(f'conv1 layer : {x.shape}')
#[4, 6, 28, 28]
x = pool(x)
print(f'pool layer : {x.shape}')
#[4, 6, 14, 14]
x = conv2(x)
print(f'conv2 layer : {x.shape}')
#[4, 16, 10, 10]
'''
class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


model = CNN()
#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=leaning_rate)

#training
for epoch in range(num_epochs):

    for i,(images,labels) in enumerate(train_loader):
            output = model(images).to(device)
            #loss
            loss = criterion(output,labels)
            #update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 16 == 0:
                 print(f'epoch {epoch+1}/{num_epochs},step {i+1}/{len(train_loader)},loss : {loss.item():.3f}')


print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
#accuracy

with torch.no_grad():
    n_samples = 0
    n_correct = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for i,(images,labels) in enumerate(test_loader):
            labels = labels.to(device)
            output = model(images).to(device)

            _,predicted = torch.max(output,1)
            n_samples += labels.size(0)
            n_correct += (predicted==labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
    





