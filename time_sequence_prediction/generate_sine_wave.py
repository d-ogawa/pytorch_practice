import numpy as np
import torch

np.random.seed(1)

T = 20
L = 1000    # data length
N = 100     # num data

x = np.empty((N, L), "int64")
x[:] = np.array(range(L) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1))
data = np.sin(1.0 * x / T).astype("float64")
torch.save(data, open("traindata.pt", "wb"))
