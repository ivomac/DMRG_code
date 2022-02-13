
import numpy as np
from scipy.sparse import csr_matrix


def get_bit(n, b):
    return ((n & 1 << b) != 0)


def swap(n, b):
    return n ^ (1 << b)


def get_bit_and_swap(n, b):
    k = 1 << b
    return ((n & k) != 0), n ^ k


def add_entry(a, v):
    a[0].append(v[0])
    a[1][0].append(v[1])
    a[1][1].append(v[2])


def state(H, n):
    from scipy.sparse.linalg import eigsh
    E, v = eigsh(H, k=(1+n), which='SA', return_eigenvectors=True)
    return E[n], v[:, n]


def Heisenberg(P, ED):
    hx = eval(P['couplings']['h_x'])
    J = P['couplings']['J'].copy()
    for i in range(len(J)):
        J[i] = eval(J[i])

    L = ED['L']

    H = ([], ([], []))

    for i in range(2**L):
        for l in range(L-1):
            b1, j = get_bit_and_swap(i, l)
            add_entry(H, (hx, i, j))
            b2, j = get_bit_and_swap(j, l+1)
            if b1 == b2:
                add_entry(H, (J[2], i, i))
                add_entry(H, (-J[1], i, j))
            else:
                add_entry(H, (-J[2], i, i))
                add_entry(H, (J[1], i, j))
            add_entry(H, (J[0], i, j))
        j = swap(i, L-1)
        add_entry(H, (hx, i, j))
    return csr_matrix(H, dtype=getattr(np, P['dtype']))


def Rydberg(P, ED):
    V = eval(P['couplings']['V'])
    Ω = eval(P['couplings']['y'])**(-6)
    Δ = eval(P['couplings']['x'])*Ω

    L = ED['L']

    H = ([], ([], []))

    for i in range(2**L):
        d_total = 0
        V_total = 0
        for l in range(L):
            b1, j = get_bit_and_swap(i, l)
            add_entry(H, (-Ω/2, i, j))
            if b1 == 1:
                d_total += 1
                for k in range(1, l+1):
                    if get_bit(i, l-k):
                        V_total += 1/k**6
        add_entry(H, (-Δ*d_total+V*V_total, i, i))
    return csr_matrix(H, dtype=getattr(np, P['dtype']))


