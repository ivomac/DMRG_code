
import numpy as np


def Heisenberg(P):
    hx = eval(P['couplings']['h_x'])
    J = P['couplings']['J'].copy()
    for i in range(len(J)):
        J[i] = eval(J[i])
    J[1] = -J[1]

    S = np.array([
        [[1, 0],
         [0, 1]],
        [[0, 1],
         [1, 0]],
        [[0, -1],
         [1, 0]],
        [[1, 0],
         [0, -1]],
    ])

    χ = sum(np.array(J) != 0)+2
    H = np.zeros((χ, χ, 2, 2), dtype=getattr(np, P['dtype']))

    H[0, 0] = np.copy(S[0])
    H[-1, -1] = np.copy(S[0])

    H[0, -1] = hx*np.copy(S[1])

    k = 1
    for i in range(len(J)):
        if J[i] != 0:
            H[0, k]  =        np.copy(S[i+1])
            H[k, -1] = J[i] * np.copy(S[i+1])
            k+=1

    for k in range(2):
        H = np.swapaxes(H, 1+k, 2+k)

    return H, np.copy(S[-1])


def Rydberg(P):
    V = eval(P['couplings']['V'])
    Ω = eval(P['couplings']['y'])**(-6)
    Δ = eval(P['couplings']['x'])*Ω

    PL = P['power_law']

    S = np.array([
        [[1, 0],
         [0, 1]],
        [[1, 0],
         [0, 0]],
        [[-Δ, -Ω/2],
         [-Ω/2, 0]],
    ])

    χ = PL['exp_terms']+2
    H = np.zeros((χ, χ, 2, 2), dtype=getattr(np, P['dtype']))

    H[0, 0] = np.copy(S[0])
    H[0, -1] = np.copy(S[2])
    H[-1, -1] = np.copy(S[0])

    if PL['exp_terms']:

        n = np.arange(0, PL['cutoff'], dtype=getattr(np, P['dtype']))

        y = (n+1)**(-6)

        def fun(k):
            f = np.zeros(PL['cutoff'], dtype=n.dtype)
            for i in range(PL['exp_terms']):
                f += k[i] * k[i+PL['exp_terms']]**n
            return f - y

        def jac(k):
            J = np.zeros((PL['cutoff'], 2*PL['exp_terms']), dtype=n.dtype)
            for i in range(PL['exp_terms']):
                J[:, i] = k[i+PL['exp_terms']]**n
                J[:, i+PL['exp_terms']] = n*k[i]*k[i+PL['exp_terms']]**(n-1)
            return J

        k0 = np.ones(2*PL['exp_terms'], dtype=n.dtype)
        from lib.fit import ls
        K = ls(PL['fit'], fun, k0, jac=jac,
               bounds=([-1.]*2*PL['exp_terms'], [1.]*2*PL['exp_terms'])).x

        for i in range(PL['exp_terms']):
            H[0, i+1]   = K[i] * np.copy(S[1])
            H[i+1, i+1] = K[i+PL['exp_terms']] * np.copy(S[0])
        for i in range(PL['exp_terms']):
            H[i+1, -1] = V * np.copy(S[1])

    else:
        H[0, 1] = np.copy(S[1])
        for i in range(PL['cutoff']-1):
            H[i+1, i+2] = np.copy(S[0])
        for i in range(1, PL['cutoff']+1):
            H[i, -1] = V / i**6 * np.copy(S[1])

    for k in range(2):
        H = np.swapaxes(H, 1+k, 2+k)

    return H, np.copy(S[1])

