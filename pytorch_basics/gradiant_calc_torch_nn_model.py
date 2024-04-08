import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

X_test = torch.tensor([5],dtype=torch.float32)

#design model
n_samples,n_features = X.shape
print(n_samples,n_features)

input_size = n_features
output_size = n_features



class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,input):
        return self.linear(input)

model = LinearRegression(input_size,output_size)

#define loss and optimizer
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

print(f'prediction before training f(5) = {model(X_test).item():.3f}' )
n_epochs = 100

for epoch in range(n_epochs):
    #forward
    y_pred = model(X)
    #loss
    l = loss(y,y_pred)
    #gradiant = backward pass
    l.backward()
    #update weights
    optimizer.step()
    #zero grads
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch - {epoch+1} ,weight = {w[0][0].item():.3f} , loss = {l:.3f}')

print(f'prediction after training f(5) = {model(X_test).item():.3f}' )