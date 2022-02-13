
import numpy as np
from scipy.linalg import svd

def TFI(L, hx, Jy, Jz):
    H = np.diag([hx] * L) + \
        np.diag([Jy] * (L-1), k=1) + \
        np.diag([Jz] * (L-1), k=-1)
    return H


def ground_state_energy(model, *i):
    U, S, V = svd(model(*i))
    return -sum(S)


