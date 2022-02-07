from torch.nn import Module, Linear
from torch.nn.functional import tanh
import torch.nn as nn
import torch.nn.functional as F
import torch
# simple mlp
# class Model(Module):
#     def __init__(self, input_shape, nb_classes, *args, **kwargs):
#         super(Model, self).__init__()
#         self.fc1 = Linear(input_shape[0], 128)
#         self.fc2 = Linear(128, nb_classes)

#     def forward(self, x):
# #         print(x.shape)
#         x = self.fc1(x)
#         x = tanh(x)
#         x = self.fc2(x)
#         return x
#compas best mlp 
class Model(Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(Model, self).__init__()
        self.fc1 = Linear(input_shape[0], 25)
        self.dropout1= nn.Dropout(0.25)
        self.fc2 = Linear(25, 75)
        self.dropout2= nn.Dropout(0.30)
        self.fc3 = Linear(75, 200)
        self.dropout3= nn.Dropout(0.30)
        self.fc4 = Linear(200, nb_classes)
        self.dropout4= nn.Dropout(0.30)

    def forward(self, x):
#         print(x.shape)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.dropout4(x)
        x = F.relu(x)
        return x
# best adult_parametrs model
# class Model(Module):
#     def __init__(self, input_shape, nb_classes, *args, **kwargs):
#         super(Model, self).__init__()
#         self.fc1 = Linear(input_shape[0], 128)
#         self.dropout1= nn.Dropout(0.08)
#         self.fc2 = Linear(128, nb_classes)
#         self.dropout2= nn.Dropout(0.13)

#     def forward(self, x):
# #         print(x.shape)
#         x = self.fc1(x)
#         x = self.dropout1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = self.dropout2(x)
#         x = F.relu(x)
#         return x

    

# class Model(nn.Module):
    
#     def __init__(self, input_shape, num_classes,*args, **kwargs):
#         super(Model,self).__init__()
#         self.num_features = input_shape[0]
#         self.num_classes = num_classes
#         self.layers = 0
        
#         self.lin1 = torch.nn.Linear(self.num_features,  150)        
#         self.lin2 = torch.nn.Linear(50, 50)        
#         self.lin3 = torch.nn.Linear(50, 50)
        
#         self.lin4 = torch.nn.Linear(150, 150) 
        
#         self.lin5 = torch.nn.Linear(50, 50)        
#         self.lin6 = torch.nn.Linear(50, 50)
#         self.lin10 = torch.nn.Linear(150, self.num_classes)
        
#         self.prelu = nn.PReLU()
#         self.dropout = nn.Dropout(0.25)

#     def forward(self, xin):
#         self.layers = 0
        
#         x = F.relu(self.lin1(xin))
#         self.layers += 1
        
#         #x = F.relu(self.lin2(x))
#         #self.layers += 1
#         for y in range(8):
#           x = F.relu(self.lin4(x)) 
#           self.layers += 1
           
#         x = self.dropout(x)
        
#         x = F.relu(self.lin10(x)) 
#         self.layers += 1
#         return x