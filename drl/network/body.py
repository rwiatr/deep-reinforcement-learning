import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap(items, n):
    if hasattr(items, '__len__'):
        expected = n - 1
        if len(items) == expected:
            return items
        else:
            raise Exception('items has len of %d, expected %d' % (len(items), expected))
    else:
        return [items] * n


def init_layer(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class FCNet(nn.Module):
    """
    l_dims = list of layer dimensions
    actv = list of activation functions
    """

    def __init__(self, l_dims=(64, 64), actv=F.relu, seed=None):
        super(FCNet, self).__init__()
        if seed:
            self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList(
            [init_layer(nn.Linear(_in, _out)) for _in, _out in zip(l_dims[:-1], l_dims[1:])]
        )
        self.activations = unwrap(actv, len(l_dims))

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation:
                x = activation(x)

        return x


def vanilla_action_fc_net(s_dim, a_dim, fc_units=(0, 400, 300, 1), actv=(F.relu, F.relu, None), action_cat=1):
    fc_units = list(fc_units)
    fc_units[0] = s_dim  # s_dim is input dimensions
    return ActionFCNet(l_dims=tuple(fc_units), actv=actv, a_dim=a_dim, action_cat=action_cat)


class ActionFCNet(nn.Module):
    """
    l_dims = list of layer dimensions
    actv = list of activation functions
    """

    def __init__(self, l_dims=(64, 64), actv=F.relu, a_dim=0, action_cat=1):
        super(ActionFCNet, self).__init__()
        self.layers = nn.ModuleList(
            [init_layer(nn.Linear(_in if idx != action_cat else _in + a_dim, _out))
             for idx, (_in, _out) in enumerate(zip(l_dims[:-1], l_dims[1:]))]
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
