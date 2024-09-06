import numpy as np
import torch


class Binance_USD_F_BTCUSDT:
    def __init__(self, normalization="zscore", hours=[], T=100, k=5):
        self.normalization = normalization
        self.hours = hours
        self.T = T
        self.k = k

        x = np.zeros((0, 0))
        y = np.zeros((0, 0))
        for hour in self.hours:
            x_loaded = np.loadtxt(
                f"C:/Users/Hallo.QQ/dev/binance-LOB/data/X/x_{self.normalization}_hour_{hour}.csv"
            )[: -self.k, :]

            y_loaded = np.loadtxt(
                f"C:/Users/Hallo.QQ/dev/binance-LOB/data/Y/y_{self.normalization}_k_{self.k}_hour_{hour}.csv"
            )

            if x.shape[0] == 0:
                x = x_loaded
                y = y_loaded
            else:
                x = np.concatenate((x, x_loaded), axis=0)
                y = np.concatenate((y, y_loaded), axis=0)

        x = x[: -(x.shape[0] % self.T)]
        x = x.reshape(int(x.shape[0] / self.T), self.T, x.shape[1])
        x = torch.from_numpy(x)
        x = x.unsqueeze(1)
        self.x = x

        y = y[: -(y.shape[0] % self.T)]
        y = y.reshape(int(y.shape[0] / self.T), self.T)
        y = torch.from_numpy(y)

        self.y = torch.zeros(y.shape[0], 1)
        for i in range(y.shape[0]):
            up = np.count_nonzero(y[i] == 1)
            down = np.count_nonzero(y[i] == -1)

            zero_percent = (100 - up - down) / 100

            if zero_percent > 0.85:
                self.y[i] = 0 + 1
            elif up > down:
                self.y[i] = 1 + 1
            else:
                self.y[i] = -1 + 1

        self.y = self.y.squeeze(1)

        self.length = len(self.y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

    def get_midprice(self):
        return []
