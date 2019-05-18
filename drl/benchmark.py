import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_min(a, n=3):
    result = np.zeros(len(a) - n + 1)
    for idx in range(0, len(a) - n + 1):
        result[idx] = np.min(a[idx:idx + n])
    return result


def transform(mode, a):
    mode, value = mode.split('=')
    if mode == 'average':
        return moving_average(a, n=int(value))
    if mode == 'min':
        return moving_min(a, n=int(value))


class ScorePlot:
    def __init__(self):
        self.scores = dict()

    def add(self, name, score):
        self.scores[name] = score

    def rem(self, name):
        self.scores.pop(name, None)

    def clear(self):
        self.scores = dict()

    def show(self, mode=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for name, score in self.scores.items():
            if mode:
                score = transform(mode, score)
            plt.plot(np.arange(len(score)), score, label=name)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
