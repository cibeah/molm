import torch
import torch.nn as nn
import torch.nn.functional as F

from initializations import init_layer

class MomentNetwork(nn.Module):
    def __init__(self, Z_dim=128, X_dim=32, H1_dim=512, H2_dim=256, H3_dim=128):
        super().__init__()
        
        self.Z_dim = 128
        self.X_dim = 32
        self.H1_dim = 512
        self.H2_dim = 256
        self.H3_dim = 128
                
        #Size-Preserving Convolutions
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=H3_dim, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(in_channels=H3_dim, out_channels=H2_dim, kernel_size=3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(in_channels=H2_dim, out_channels=H1_dim, kernel_size=3, stride=1, padding=1)
        init_layer(self.conv11)
        init_layer(self.conv21)
        init_layer(self.conv31)
        
        
        #Stride 2 Convolutions
        self.conv12 = nn.Conv2d(in_channels=H3_dim, out_channels=H3_dim, kernel_size=3, stride=2, padding=1)
        self.conv22 = nn.Conv2d(in_channels=H2_dim, out_channels=H2_dim, kernel_size=3, stride=2, padding=1)
        self.conv32 = nn.Conv2d(in_channels=H1_dim, out_channels=H1_dim, kernel_size=3, stride=2, padding=1)
        init_layer(self.conv12)
        init_layer(self.conv22)
        init_layer(self.conv32)
        
        #Output Linear Layer
        self.fc = nn.Linear(in_features=H1_dim * 4 * 4, out_features=1)
        
        #Leaky ReLus:
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
   # def extra_repr(self):
   #     return "Generalized Method of Moments"
        
    def forward(self, t):
        h = []
        
        # (1) input layer
        t = t
        #h.append(t)
        #print(t.shape)
        
        #(1) size-preserving + stride-2
        t = self.conv11(t)
        t = self.lrelu(t)
        h.append(t.reshape(-1, self.H3_dim * 32 * 32))
        #print(t.shape)
        t = self.conv12(t)
        #print(t.shape)
        t = self.lrelu(t)
        h.append(t.reshape(-1, self.H3_dim * 16 * 16))
    
        
        #(2) size-preserving + stride-2
        t = self.conv21(t)
        t = self.lrelu(t)
        h.append(t.reshape(-1, self.H2_dim * 16 * 16))
        #print(t.shape)
        t = self.conv22(t)
        #print(t.shape)
        t = self.lrelu(t)
        h.append(t.reshape(-1, self.H2_dim * 8 * 8))
        
        #(3) size-preserving + stride-2
        t = self.conv31(t)
        t = self.lrelu(t)
        h.append(t.reshape(-1, self.H1_dim * 8 * 8))
        #print(t.shape)
        t = self.conv32(t)
        #print(t.shape)
        t = self.lrelu(t)
        h.append(t.reshape(-1, self.H1_dim * 4 * 4))
        
        #(4) reshape + Linear Layer + sigmoid activation
        t = t.reshape(-1, self.H1_dim * 4 * 4)
        t = self.fc(t)
        prob = torch.sigmoid(t)
       # print(t.shape)
        
        return prob, t, h

    def get_gradients(self, fn):
        fn.backward(retain_graph=True)
        grads = []
        for param_tensor in self.parameters():
            grads.append(param_tensor.grad.reshape(1,-1))
        
        gradients = torch.cat(
            grads,
            dim =1
        )        
        return gradients

    def get_moment_vector(self, x, size, weights=1e-4, detach=False):
        _, output, hidden = self.forward(x)
        
        mean_output = output.mean()
        grad_monet = self.get_gradients(mean_output)
        grad_monet = (grad_monet / size).squeeze()
        if detach:
            hidden = [h.detach() for h in hidden]
            grad_monet = grad_monet.detach()
        
        activations = torch.cat(
            hidden,
            dim = 1
        )
        mean_activations = activations.mean(0) * weights
        mean_input = x.mean(0).reshape(-1)
        moments = torch.cat(
            [grad_monet, mean_input, mean_activations],
            dim =0
        )
        
        return moments