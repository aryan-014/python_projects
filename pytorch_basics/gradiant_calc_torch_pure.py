import torch

X = torch.tensor([1,2,3,4,5],dtype=torch.float32)
y = torch.tensor([2,4,6,8,10],dtype=torch.float32)

w =  torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

def forward(X):
    return w*X

def loss(y,y_prediction):
    return ((y_prediction-y)**2).mean()



learning_rate = 0.01

print("prediction before training f(5) = ",forward(5).item())
n_epochs = 20

for epoch in range(n_epochs):
    #forward
    y_pred = forward(X)
    #loss
    l = loss(y,y_pred)
    #gradiant
    l.backward()
    #update weights
    with torch.no_grad():
        w -= learning_rate*w.grad
    
    w.grad.zero_()

    if epoch % 2 == 0:
        print(f'epoch - {epoch+1} ,weight = {w:.3f} , loss = {l:.3f}')

print("prediction after training f(5) = ",forward(5).item())