import torch
import torch.nn as nn

X = torch.tensor([1,2,3,4,5],dtype=torch.float32)
y = torch.tensor([2,4,6,8,10],dtype=torch.float32)

w =  torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

def forward(X):
    return w*X

learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w],lr=learning_rate)

print("prediction before training f(5) = ",forward(5).item())
n_epochs = 10

for epoch in range(n_epochs):
    #forward
    y_pred = forward(X)
    #loss
    l = loss(y,y_pred)
    #gradiant = backward pass
    l.backward()
    #update weights
    optimizer.step()
    #zero grads
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch - {epoch+1} ,weight = {w:.3f} , loss = {l:.3f}')

print("prediction after training f(5) = ",forward(5).item())