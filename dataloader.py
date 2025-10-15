import numpy as np

class DataLoader:
    def __init__(self, X, Y, batch_size=1, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        N = self.X.shape[0]
        indices = np.arange(N)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, N, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield self.X[batch_idx], self.Y[batch_idx]