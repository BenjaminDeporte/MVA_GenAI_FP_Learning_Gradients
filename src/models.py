# WORK IN PROGRESS


import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class CMGN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super(CMGN, self).__init__()
        self.num_layers = num_layers
        self.sigma_layers = nn.ModuleList([nn.Sigmoid() for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_layers)])
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim)) # Share the weight matrix W across all layers
        self.V = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.bL = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):

        Wx = F.linear(x, self.W)
        z = Wx + self.biases[0] 
        for i in range(1,self.num_layers):
            z = Wx + self.sigma_layers[i-1](z) + self.biases[i]

        VTx = F.linear(x, self.V.t())
        VTVx = F.linear(VTx, self.V)
        Wsigma = F.linear(self.sigma_layers[self.num_layers-1](z), self.W.T)

        output = Wsigma + VTVx + self.bL
        return output
    

class MMGN(nn.Module):
    def __init__(self, input_dim, output_dim, num_modules, hidden_dim):
        super(MMGN, self).__init__()
        self.num_modules = num_modules
        self.W_list = nn.ParameterList([nn.Parameter(torch.randn(input_dim, hidden_dim)) for _ in range(num_modules)])
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_modules)])
        self.activation = nn.Tanh()

        self.V = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.a = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        def log_cosh(x):
          return torch.sum( torch.log(torch.cosh(x)) )

        sum = 0
        for i in range(self.num_modules):
            Wx = F.linear(x, self.W_list[i].T)
            z = Wx + self.biases[i]
            Wsigma = F.linear(self.activation(z), self.W_list[i])
            sum += log_cosh(z) * Wsigma
        VTx = F.linear(x, self.V)
        VTVx = F.linear(VTx, self.V.t())

        output = self.a + VTVx + sum 
        return output
