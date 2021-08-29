import numpy as np


class MovAvg:
    """Moving Average with Standard Deviation"""
    def __init__(self, max_size=100):
        self.maxsize = max_size
        self.cache = np.zeros(max_size)

        self.sum = 0.0
        self.sq_sum = 0.0
        self.size = 0
        self.pointer = 0

    def push(self, item):
        if self.size == self.maxsize:
            a = self.cache[self.pointer]

            self.sum -= a
            self.sq_sum -= a ** 2
        else:
            self.size += 1

        self.cache[self.pointer] = item
        self.pointer = (self.pointer + 1) % self.maxsize

        self.sum += item
        self.sq_sum += item ** 2

    def get(self):
        if self.size == 0:
            return 0

        return self.sum / self.size

    def std(self):
        if self.size == 0:
            return 0

        avg = self.sum / self.size
        return np.sqrt((self.sq_sum - self.sum * avg) / self.size)

    def __repr__(self):
        return "Mean: {:.3f} Std {:.3f}".format(self.get(), self.std())
