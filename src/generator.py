import torch
import torch.nn as nn
import torch.nn.functional as F

from initializations import init_layer

class Generator(nn.Module):
    
    def __init__(self, Z_dim=128, X_dim=32, H1_dim=512, H2_dim=256, H3_dim=128):
        super().__init__()
        
        self.dims = [Z_dim, H1_dim, H2_dim, H3_dim, X_dim]
        
        #Fully Connected Linear Layers, no bias for an equivalent to a matmul
        self.fc1 = nn.Linear(in_features=Z_dim, out_features=H1_dim*4*4, bias=False)
        init_layer(self.fc1)
        
        #Transposed Convolutions
        self.convt1 = nn.ConvTranspose2d(H1_dim, H2_dim, kernel_size=4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(H2_dim, H3_dim, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(H3_dim, 3, kernel_size=4, stride=2, padding=1)
        init_layer(self.convt1)
        init_layer(self.convt2)
        init_layer(self.convt3)
        
        #Batch Normalizations
        self.bnorm1 = nn.BatchNorm2d(H1_dim)
        self.bnorm2 = nn.BatchNorm2d(H2_dim)
        self.bnorm3 = nn.BatchNorm2d(H3_dim)
        init_layer(self.bnorm1, True)
        init_layer(self.bnorm2, True)
        init_layer(self.bnorm3, True)
 
    def forward(self, t):
        # (1) input layer
        t = t
        
        # (2) Latent space vector => project and reshape
        t = self.fc1(t)
        t = t.reshape(-1, self.dims[1], 4, 4)
        t = F.relu(t)
        t = self.bnorm1(t)
        #print(t.shape)
        
        # (3) First fractionnally-strided conv layer
        t = self.convt1(t)
        t = F.relu(t)
        t = self.bnorm2(t)
        #print(t.shape)
        
        # (4) Second fractionnally-strided conv layer
        t = self.convt2(t)
        t = F.relu(t)
        t = self.bnorm3(t)
        #print(t.shape)
        
        # (5) Last transposed conv layer => image generation
        t = self.convt3(t)
        t = torch.tanh(t)
         
        return t