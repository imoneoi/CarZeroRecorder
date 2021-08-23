import multiprocessing as mp
import ctypes

import numpy as np


class SharedArray:
    def __init__(self, shape: tuple, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype

        self.arr = mp.Array(ctypes.c_uint8, int(np.prod(shape) * np.dtype(dtype).itemsize), lock=False)

    def get(self):
        return np.frombuffer(self.arr, dtype=self.dtype).reshape(self.shape)

    def __repr__(self):
        return "<SharedArray shape={}, dtype={}>".format(self.shape, self.dtype)

