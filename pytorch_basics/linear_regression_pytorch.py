import torch
import numpy as np
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

#prepare data
X_numpy,y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

X= torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

#resize X_numpy
print(X.shape)
print("y shape : ",y.shape)
y = y.view(y.shape[0],1)#row,col

#1)model
n_samples,n_features = X.shape

input_size = n_features
output_size = 1

model = nn.Linear(input_size,output_size)

#2)loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


#3)training loop
n_epochs = 100

for epoch in range(n_epochs):
    #forward
    y_pred = model(X)
    #loss
    l = criterion(y_pred,y)
    #gradiants
    l.backward()
    #update weights
    optimizer.step()
    #zero gradiant for next iteration
    optimizer.zero_grad()

    #print for every 10 epochs
    if (epoch+1) % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch - {epoch+1} ,weight = {w[0][0].item():.3f} , loss = {l:.3f}')

    

#prediction
# predicted = model(X).detach().numpy()
# plt.plot(X_numpy,y_numpy,'ro')#real data with red color
# plt.plot(X_numpy,predicted,'b')#predicted data with red color
# plt.show()


