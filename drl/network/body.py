import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def unwrap(items, n):
    if hasattr(items, '__len__'):
        expected = n - 1
        if len(items) == expected:
            return items
        else:
            raise Exception('items has len of %d, expected %d' % (len(items), expected))
    else:
        return [items] * n


class LayerInitializer:
    def init_layers(self, layers):
        raise NotImplemented()


class OrthogonalInitializer(LayerInitializer):
    def init_layer(self, layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer

    def init_layers(self, layers):
        return [self.init_layer(layer) for layer in layers]


class UInitializer(LayerInitializer):
    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return -lim, lim

    def init_layers(self, layers):
        for layer in layers[:-1]:
            layer.weight.data.uniform_(*self.hidden_init(layer))
            yield layer
        layer = layers[-1]
        layer.weight.data.uniform_(-3e-3, 3e-3)
        yield layer


class FCNet(nn.Module):
    """
    l_dims = list of layer dimensions
    actv = list of activation functions
    """

    def __init__(self, l_dims=(64, 64), actv=F.relu, seed=None, layer_initializer=OrthogonalInitializer()):
        super(FCNet, self).__init__()
        if seed:
            self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList(
            list(layer_initializer.init_layers(
                [nn.Linear(_in, _out) for _in, _out in zip(l_dims[:-1], l_dims[1:])]))
        )
        self.activations = unwrap(actv, len(l_dims))

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation:
                x = activation(x)

        return x


class ActionFCNet(nn.Module):
    """
    l_dims = list of layer dimensions
    actv = list of activation functions
    """

    def __init__(self, l_dims=(64, 64), actv=F.relu, a_dim=0, action_cat=1, seed=None,
                 layer_initializer=UInitializer()):
        super(ActionFCNet, self).__init__()
        if seed:
            self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList(
            layer_initializer.init_layers([nn.Linear(_in if idx != action_cat else _in + a_dim, _out)
                                           for idx, (_in, _out) in enumerate(zip(l_dims[:-1], l_dims[1:]))])
        )
        self.activations = unwrap(actv, len(l_dims))
        self.action_cat = action_cat

    def forward(self, x, action):
        for idx, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            if idx is self.action_cat:
                x = torch.cat((x, action), dim=1)

            x = layer(x)
            if activation:
                x = activation(x)

        return x
