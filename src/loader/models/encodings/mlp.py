import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            if i < len(layer_sizes) - 2:
                self.layers.append(activation)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x