import torch.nn as nn
import torch.nn.functional as F


def unwrap(items, n):
    if hasattr(items, '__len__'):
        if len(items) == n:
            return items
        else:
            raise Exception('items has len of %d, expected %d'.format(len(items), n))
    else:
        return [items] * n


def init_layer(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class FCNet(nn.Module):

    def __init__(self, dims=(64, 64), fs=F.relu):
        super(FCNet, self).__init__()
        self.layers = zip(
            [init_layer(nn.Linear(_in, _out)) for _in, _out in zip(dims[:-1], dims[1:])],
            unwrap(fs, len(dims))
        )

    def forward(self, x):
        for l, f in self.layers:
            x = f(l(x))
        return x
