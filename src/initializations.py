import torch.nn as nn

def init_layer(layer, std=0.02, bias=False):
    nn.init.normal_(layer.weight.data, mean=0, std=std)
    if bias:
        nn.init.constant_(layer.bias.data, val=0)