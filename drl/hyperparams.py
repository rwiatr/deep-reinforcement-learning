import numpy as np


class HyperParams:
    """"""

    def __str__(self):
        return str(self.__dict__)


def default():
    params = HyperParams()
    params.batch_size = 128
    params.buffer_size = int(1e6)
    params.seed = 1
    params.wd_a = 0
    params.wd_c = 0.0001
    params.lr_c = 1e-3
    params.lr_a = 1e-4
    params.batch_size = 128
    params.gamma = 0.99
    params.tau = 1e-3
    return params


def hyper_space_sc_rec(map, keys):
    if len(keys) == 1:
        return [{keys[0]: value} for value in map[keys[0]]]
    rest = hyper_space_sc_rec(map, keys[1:])
    return [dict(items, **{keys[0]: value}) for items in rest for value in map[keys[0]]]


def hyper_space_ns(params={'lr': [5e-8, 5e-1]}):
    all_keys = list(params.keys())
    for data in hyper_space_sc_rec(params, all_keys):
        properties = default()
        for key in data:
            setattr(properties, key, data[key])
        yield properties


def hyper_space(params={'lr': [5e-8, 5e-1]}, n=5, d=0):
    property_matrix = np.zeros([len(params), n])
    for idx, key in enumerate(params):
        f, t = params[key]
        np.clip(np.linspace(f, t, n, endpoint=True), f, t, out=property_matrix[idx])

    displacement = (np.random.rand(n - 2) - .5) * 2 * d
    property_matrix[:, 1:-1] -= property_matrix[:, 1:-1] * displacement

    index_vector = np.zeros(len(params), dtype='int64')
    while True:
        properties = HyperParams()
        for idx, key in enumerate(params):
            value = property_matrix[idx, index_vector[idx]]
            if key in ['buffer_size', 'batch_size', 'update_every', 'fc_size']:
                value = int(value)
            setattr(properties, key, value)
        yield properties

        k = 0
        while True:
            index_vector[k] = (index_vector[k] + 1) % n
            if index_vector[k] == 0:
                k += 1
            else:
                break

            if k == len(params):
                return
