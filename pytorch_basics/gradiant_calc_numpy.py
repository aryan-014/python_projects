import numpy as np

X = np.array([1,2,3,4,5],dtype=np.float32)
y = np.array([2,4,6,8,10],dtype=np.float32)

w = 0.0

def forward(X):
    return w*X

def loss(y,y_prediction):
    return ((y_prediction-y)**2).mean()

def gradiant(x,y,y_prediction):
    return np.dot(2*x,y_prediction-y).mean()


learning_rate = 0.01

print("prediction before training f(5) = ",forward(5))
n_epochs = 20

for epoch in range(n_epochs):
    #forward
    y_pred = forward(X)
    #loss
    l = loss(y,y_pred)
    #gradiant
    dw = gradiant(X,y,y_pred)
    #update weights
    w -= learning_rate*dw

    # if epoch % 1 == 0:
        # print(f'epoch - {epoch+1} ,weight = {w:.3f} , loss = {l:.3f}')

print("prediction after training f(5) = ",forward(5))



