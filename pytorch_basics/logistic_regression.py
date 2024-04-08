import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#data preparation

bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target


print("X.shape -> ",X.shape,", y.shape ->",y.shape)

n_samples,n_features = X.shape

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

#scale
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

print("X_train : ",X_train[0:2])

y_train = y_train.view(y_train.shape[0],1)#y has only rows
print("y_train : ",y_train[0:5])
y_test = y_test.view(y_test.shape[0],1)


#model

class LogisticRegression(nn.Module):

    def __init__(self,n_features):
        super().__init__()
        self.linear = nn.Linear(n_features,1)
    
    def forward(self,input):
        predicted = torch.sigmoid(self.linear(input))
        return predicted
    

model = LogisticRegression(n_features)

#loss and optimizer

learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#training

n_epochs = 100

for epoch in range(n_epochs):
    #forward
    y_pred = model(X_train)
    #loss
    loss = criterion(y_pred,y_train)
    #cal. gradiant - backward
    loss.backward()
    #update weights
    optimizer.step()
    optimizer.zero_grad()

    # if (epoch+1) % 10 == 0:
    #     [w,b] = model.parameters()
    #     print(f'epoch - {epoch+1} ,weight = {w[0][0].item():.3f} , loss = {loss:.3f}')

with torch.no_grad():#do not include gradiant calcualtion
    predicted = model(X_test)
    predicted_cls = predicted.round()#give 0 and 1
    accuracy = predicted_cls.eq(y_test).sum()/round(y_test.shape[0])
    print("accuracy : ",accuracy)
    # plt.plot(X_test[0],y_test,'ro')#real data with red color
    # plt.plot(X_test[0],predicted,'b')#predicted data with red color
    # plt.show()


