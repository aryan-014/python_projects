import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    

model = Model(n_input_features=6)
# for param in model.parameters():
#     print(param)


FILE = "model_only_state.pth"
#method 1 - save all

#1. save the full model
# torch.save(model,FILE)
#2. load the model
# model = torch.load(FILE)
# for param in model.parameters():
#     print(param)


#method 2 - save only state dict
# torch.save(model.state_dict(), FILE)
# load_model = Model(n_input_features=6)
# load_model.load_state_dict(torch.load(FILE))
# load_model.state_dict()
# load_model.eval()


###########3.load checkpoint#####################

# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# checkpoint = {
# "epoch": 90,
# "model_state": model.state_dict(),
# "optim_state": optimizer.state_dict()
# }
# print(optimizer.state_dict())
FILE = "checkpoint.pth"
# torch.save(checkpoint, FILE)

#load checkpoint

checkpoint = torch.load(FILE)
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
print(optimizer.state_dict())
epoch = checkpoint['epoch']

model.eval()


# Remember that you must call model.eval() to set dropout and batch normalization layers 
# to evaluation mode before running inference. Failing to do this will yield 
# inconsistent inference results. If you wish to resuming training, 
# call model.train() to ensure these layers are in training mode.

""" SAVING ON GPU/CPU 

# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Note: Be sure to use the .to(torch.device('cuda')) function 
# on all model inputs, too!

# 3) Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)

# This loads the model to a given GPU device. 
# Next, be sure to call model.to(torch.device('cuda')) to convert the model’s parameter tensors to CUDA tensors
"""


