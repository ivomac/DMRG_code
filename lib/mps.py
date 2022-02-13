
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from math import sqrt, floor, ceil

def w(A, i):
    if i == +1:
        return 0, A-1
    elif i == -1:
        return A-1, 0

class MPS(list):

    def __init__(mps, P):
        mps.s, mps.b = [None]*2
        mps.set_params(P)

        super().__init__([])

        mps.init()

    @property
    def L(mps):
        return len(mps)

    @property
    def norm_type(mps):
        if mps.b == mps.L-1:
            return +1
        elif mps.b == 0:
            return -1
        return

    @property
    def max_D(mps):
        lst = map(lambda m: m.shape[::2], mps)
        return max([item for sublist in lst for item in sublist])


    def set_params(mps, P, full=True):
        from lib import mpo

        setattr(mps, 'P', P['Simulation'])
        setattr(mps, 'System', P['System'])
        setattr(mps, 'Lanczos', P['Lanczos'])
        setattr(mps, 'Initialization', P['Initialization'])
        setattr(mps, 'dtype', getattr(np, P['Simulation']['dtype']))
        if full:
            setattr(mps, 'm', dict())
            setattr(mps, 'sweeps', 0)

        setattr(mps, 'force_svd', False)

        for t in ['H', 'O', 'd', 'K']:
            setattr(mps, t, None)
        mps.d = P['Physical_Bond_Dimension']
        mps.H, mps.O = getattr(mpo, mps.System)(mps.P)
        return

    def append_measure(mps, tp, var):
        if mps.m.get(tp) is None:
            mps.m[tp] = [var]
        else:
            mps.m[tp].append(var)
        return

    def init(mps):
        from lib import ED
        if mps.Initialization['ED']['use']:
            E_ED, psi = ED.state(
                getattr(ED, mps.System)(mps.P, mps.Initialization['ED']),
                mps.P['state']
            )
            mps.psi_to_mps(psi)
        else:
            mps.init_random()
        return

    def SVD(mps, l):
        D = mps.P['svd']['cutoff']

        U, s, V = la.svd(mps[l], full_matrices=False, check_finite=False)

        if mps.P['svd']['minimum_value']:
            k = None
            for i, n in enumerate(s):
                if n < mps.P['svd']['minimum_value']:
                    k = i
                    break
            if k is not None:
                D = min(k, D)

        U = U[:, :D]
        V = V[:D, :]
        s = s[:D]

        if s.shape[0] == mps.P['svd']['cutoff']:
            mps.append_measure('minimum singular value', s[-1])

        nrm = la.norm(s)
        s = s/nrm
        return U, np.diag(s), V

    def QR(mps, k, i):
        D = mps.P['svd']['cutoff']
        M = mps[k]
        if i == +1:
            M.shape = (M.shape[0]*mps.d, -1)
            if (D and min(M.shape) > D) or mps.force_svd:
                Q, s, V = mps.SVD(k)
                R = np.tensordot(s, V, axes=1)
            else:
                Q, R = la.qr(M, mode='economic')
            Q.shape = (-1, mps.d, Q.shape[-1])
        elif i == -1:
            M.shape = (-1, M.shape[-1]*mps.d)
            if (D and min(M.shape) > D) or mps.force_svd:
                U, s, Q = mps.SVD(k)
                R = np.tensordot(U, s, axes=1)
            else:
                R, Q = la.rq(M, mode='economic')
            Q.shape = (Q.shape[0], mps.d, -1)
        return Q, R

    def init_random(mps):
        for j in range(mps.Initialization['random']['L']):
            mps.insert(0, None)
        rng = np.random.default_rng()
        D = mps.P['svd']['cutoff']
        for k in range(mps.L):
            d_left  = min(D, k, mps.L-k)
            d_right = min(D, k+1, mps.L-k-1)
            mps[k] = rng.random((mps.d**d_left, mps.d, mps.d**d_right)).astype(mps.dtype)
        mps.b = 0
        mps.move_border_to_edge(closest=False)
        return

    def psi_to_mps(mps, psi, i=-1):

        for j in range(mps.Initialization['ED']['L']):
            mps.insert(0, None)

        Ls, Le = w(mps.L, i)

        mps[Ls] = psi
        mps[Ls].shape = (i, -i)
        for n in range(Ls, Le, i):
            mps[n], mps[n+i] = mps.QR(n, i)

        kl, kr = w(mps.d, -i)
        mps[Le].shape = (kl+1, mps.d, kr+1)
        mps.b = Le
        mps.normalize()
        return mps

    def to_psi(mps):
        psi = mps[0]
        for n in range(1, mps.L):
            psi = np.tensordot(psi, mps[n], axes=1)
        psi = np.trace(psi, axis1=0, axis2=-1)
        psi.shape = (mps.d**mps.L)
        return psi

    def tensor_norm_type(mps, i):
        nm = mps.norm_type
        if nm:
            return nm
        elif mps.b > i:
            return +1
        elif mps.b < i:
            return -1
        return

    def contract(mps, n, *S, axes='full'):

        ax = {
            'full': ([0, 1, -1], [0, -1]),
            -1:     ([1, -1],    [1, -1]),
            +1:     ([0, 1],     [0,  2]),
        }
        ax0, ax1 = ax[axes]
        if S:
            ax0.remove(1)

        m = np.tensordot(mps[n].conj(), mps[n], axes=(ax0, ax0))

        if S:
            return np.tensordot(S[0], m, axes=([0, -1], ax1))
        else:
            return m

    def tensor_product(mps, *t):
        T = t[0]
        for i in range(1, len(t)):
            T = np.tensordot(T, t[i], axes=0)
        return np.array(T, dtype=mps.dtype)

    def H_squared(mps):
        he = np.eye(mps.H.shape[-1])
        K = mps.tensor_product([1.], he[:, -1], he[:, -1], [1.])
        K_end = mps.tensor_product([1.], he[0, :], he[0, :], [1.])

        for n in range(mps.L-1, -1, -1):
            K = np.tensordot(mps[n], K, axes=(-1, -1))
            K = np.tensordot(mps.H, K, axes=((-2, -1), (1, -1)))
            K = np.tensordot(mps.H, K, axes=((-2, -1), (1, -1)))
            K = np.tensordot(mps[n].conj(), K, axes=((-2, -1), (1, -1)))
        return np.tensordot(K_end, K, axes=((0, 1, 2, 3), (0, 1, 2, 3)))

    def extend_sidetensor(mps, n, T, i, mpo=True):
        k = {-1: -1, +1: 0}[i]
        ax = ([-2-k, 0], [1, k])
        K = np.tensordot(T, mps[n].conj(), axes=(0, k))
        if mpo:
            K = np.tensordot(K, mps.H, axes=ax)
        K = np.tensordot(K, mps[n], axes=ax)
        return K

    def build_sidetensors(mps):
        if mps.K is None:
            mps.K = [None] * (mps.L + 2)
            he = np.eye(mps.H.shape[-1])
            mps.K[0] = mps.tensor_product([1.,], he[0, :], [1.,])
            mps.K[-1] = mps.tensor_product([1.,], he[:, -1], [1.,])

        for n in range(0, ceil(mps.b)):
            if mps.K[n+1] is None:
                mps.K[n+1] = mps.extend_sidetensor(n, mps.K[n], +1)
        for n in range(mps.L-1, floor(mps.b), -1):
            if mps.K[n+1] is None:
                mps.K[n+1] = mps.extend_sidetensor(n, mps.K[n+2], -1)

        return

    def normalize(mps):
        nm = mps.contract(mps.b)
        mps[mps.b] = mps[mps.b]/sqrt(abs(nm))
        return nm

    def move_border(mps, i):
        if abs(i) == +1:
            i /= 2
            mps.move_border(i)
        elif i == 0:
            return 0

        if 2*mps.b % 2:
            mps.b = int(mps.b + i)
            if mps.K is not None:
                mps.K[mps.b+1] = None
            if i == +1/2:
                mps[mps.b] = np.tensordot(mps.s, mps[mps.b], axes=1)
            elif i == -1/2:
                mps[mps.b] = np.tensordot(mps[mps.b], mps.s, axes=1)
            mps.s = None
        else:
            mps[mps.b], mps.s = mps.QR(mps.b, np.sign(i))
            mps.b += i
            if mps.K is not None:
                mps.build_sidetensors()
        return 0

    def move_border_to(mps, k):
        while k != mps.b:
            mps.move_border(np.sign(k - mps.b)/2)
        return 0

    def move_border_to_edge(mps, closest=True):
        i = 0 if (mps.b > mps.L-1/2)^closest else mps.L-1
        mps.move_border_to(i)
        return 0

    def dmrg_update(mps, i=0, dual=True):

        l = int(mps.b-1/2) if 2*mps.b % 2 else min(mps.b, mps.b+i)
        mps_shape = (mps.K[l].shape[0],) + (mps.d,)*(1+dual) + (mps.K[l+2+dual].shape[0],)
        mat_shape = np.prod(mps_shape)

        def matrix_vector_product(v):
            v.shape = mps_shape
            M = np.tensordot(v, mps.K[l+2+dual], axes=(-1, -1))
            M = np.tensordot(mps.H, M, axes=((-2, -1), (-3, -1)))
            if dual:
                M = np.tensordot(mps.H, M, axes=((-2, -1), (-2, 0)))
            M = np.tensordot(mps.K[l], M, axes=((1, -1), (0, -2)))
            M.shape = (-1,)
            v.shape = (-1,)
            return M

        A = sla.LinearOperator((mat_shape, mat_shape), matvec=matrix_vector_product, dtype=mps.dtype)

        if mps[l] is None:
            ops = {}
        else:
            if dual:
                st = np.tensordot(mps[l], mps[l+1], axes=1)
            else:
                st = mps[l]
            st.shape = (-1,)
            ops = {'v0': st}

        def eig(A, **kwargs):
            return sla.eigsh(
                A, which='SA',
                k=(1+mps.P['state']),
                ncv=mps.Lanczos['vectors'],
                maxiter=mps.Lanczos['max_iterations'],
                tol=mps.Lanczos['precision'],
                **kwargs)

        En, m = eig(A, **ops)

        mps[l] = m[:, mps.P['state']]
        En = En[mps.P['state']]

        if dual:
            mps[l].shape = (np.prod(mps_shape[:2]), -1)
            mps[l], mps.s, mps[l+1] = mps.SVD(l)
            mps[l].shape = (mps_shape[0], mps_shape[1], -1)
            mps[l+1].shape = (-1, mps_shape[-2], mps_shape[-1])
            mps.b += i/2
            mps.build_sidetensors()
            mps.move_border(i/2)
        else:
            mps[l].shape = mps_shape
        return En

    def resize(mps, initial_same, restarted):

        def truncate(mps, D):
            m = mps.d
            mps[0]  = mps[ 0][:1, :, :m]
            mps[-1] = mps[-1][:m, :, :1]
            for i in range(1, mps.L//2):
                m *= mps.d
                mps[i] = mps[i][:mps[i-1].shape[-1], :, :min(m, D)]
                mps[-i-1] = mps[-i-1][:min(m, D), :, :mps[-i].shape[0]]

            if mps.L % 2:
                i = mps.L//2
                mps[i] = mps[i][:mps[i-1].shape[-1], :, :mps[i+1].shape[0]]
            else:
                i = mps.L//2-1
                lb = min(mps[i].shape[-1], mps[i+1].shape[0])
                mps[i] = mps[i][:, :, :lb]
                mps[i+1] = mps[i+1][:lb, :, :]

            mps.move_border_to_edge()
            mps.force_svd = True
            mps.move_border_to_edge(closest=False)
            mps.force_svd = False
            mps.normalize()
            return

        def grow(mps, l, dual=True):
            for i in range(1+dual):
                mps.insert(l, None)
                mps.K.insert(l+1, None)
            mps.b += 1
            E = mps.dmrg_update(dual=dual)
            mps.append_measure('growth energy density', E/mps.L)
            return E

        cutoff = mps.Initialization['Cutoff']

        if mps.L == mps.P['L']:
            truncate(mps, mps.P['svd']['cutoff'])
            if restarted:
                truncate(mps, cutoff['restarted'])
            elif initial_same:
                truncate(mps, cutoff['initial_same'])
            else:
                truncate(mps, cutoff['initial_not_same'])
            return

        if mps.L % 2:
            mps.move_border_to(0)
            del mps[mps.L//2]

        if mps.L > mps.P['L']:
            mps.move_border_to(0)
            initial_size = mps.L

            while mps.L > mps.P['L']:
                del mps[-1]
                del mps[0]

        truncate(mps, mps.P['svd']['cutoff'])
        truncate(mps, cutoff['initial_not_same'])

        l = mps.L//2
        mps.move_border_to(l-1/2)

        mps.build_sidetensors()

        real_D = mps.P['svd']['cutoff']
        mps.P['svd']['cutoff'] = cutoff['initial_not_same']

        while mps.L < mps.P['L'] - 1:
            E = grow(mps, l)
            l += 1

        mps.P['svd']['cutoff'] = real_D

        if mps.L == mps.P['L'] - 1:
            E = grow(mps, l, dual=False)

        return

    @property
    def Energy(mps):
        mps.build_sidetensors()
        if 2*mps.b % 2:
            mps.move_border(1/2)
        n = mps.b
        E = np.tensordot(mps.K[n], mps[n].conj(), axes=(0, 0))
        E = np.tensordot(E, mps.H, axes=([0, 2], [0, 1]))
        E = np.tensordot(E, mps[n], axes=([0, -2], [0, 1]))
        E = np.tensordot(E, mps.K[n+2], axes=([0, 1, 2], [0, 1, 2]))
        return E

    def save_energy_data(mps, E):
        mps.append_measure('energy density', E/mps.L)
        mps.append_measure('energy density squared', mps.H_squared()/mps.L**2)
        mps.append_measure('energy density error',
            sqrt(abs(mps.m['energy density squared'][-1] - mps.m['energy density'][-1]**2)))
        mps.append_measure('relative energy density error',
            abs(mps.m['energy density error'][-1]/mps.m['energy density'][-1]))
        return

    def converged(mps, initial_is_same=False):
        if not mps.m.get('relative energy density error', None):
            mps.save_energy_data(mps.Energy)

        conv = mps.m['relative energy density error'][-1] < mps.P['dmrg']['precision'] and (
            mps.sweeps >= mps.P['dmrg']['min_sweeps'] or initial_is_same
        )
        if conv:
            mps.K = None
        return conv

    def dmrg_sweep(mps):

        mps.move_border_to_edge()
        mps.normalize()
        i = -mps.norm_type

        mps.build_sidetensors()

        while 1:
            mps.save_energy_data(
                mps.dmrg_update(i=i)
            )

            if mps.norm_type:
                mps.normalize()
                mps.sweeps += 1
                return
        return

    def one_point(mps, **P):

        mps.move_border_to_edge()
        mps.normalize()

        i = -mps.norm_type
        Ls, Le = w(mps.L, i)

        m = np.zeros(mps.L, dtype=mps.dtype)

        of = (i-1)//2
        for n in range(Ls, Le, i):
            k = mps.contract(n, mps.O, axes=i)
            m[n] = np.trace(k)
            mps.move_border(i)

        m[Le] = mps.contract(Le, mps.O)

        mps.m = {**mps.m, 'one-point': m, }
        return

    def nn_two_point(mps, **P):
        mps.move_border_to_edge()
        mps.normalize()

        i = -mps.norm_type
        Ls, Le = w(mps.L, i)

        nn =   np.zeros(mps.L-1, dtype=mps.dtype)
        nn_c = np.zeros(mps.L-1, dtype=mps.dtype)
        s =    np.zeros(mps.L-1, dtype=mps.dtype)

        of = (i-1)//2
        for n in range(Ls, Le, i):
            k = mps.contract(n, mps.O, axes=i)
            j = mps.contract(n+i, mps.O, axes=-i)
            nn[n+of] = np.tensordot(j, k, axes=([0, 1], [0, 1]))
            mps.move_border(i/2)
            K = la.svd(mps.s, compute_uv=False, check_finite=False)
            s[n+of] = -sum(K**2 * np.log2(K**2))
            mps.move_border(i/2)

        m = mps.m['one-point']
        for n in range(Ls, Le, i):
            nn_c[n+of] = nn[n+of] - m[n]*m[n+i]

        mps.m = {**mps.m,
            'nearest neighbor two-point': nn,
            'nearest neighbor two-point correlation': nn_c,
            'entanglement entropy': s,
        }
        return

    def two_point(mps, **P):

        l = mps.L//2

        m_op = mps.m['one-point']

        mps.move_border_to(l)
        T = mps.contract(l, mps.O, axes=-1)

        c = np.zeros(l, dtype=mps.dtype)
        C = np.zeros(l, dtype=mps.dtype)

        for n in range(l-1, -1, -1):
            k = mps.contract(n, mps.O, axes=+1)
            c[n] = np.tensordot(k, T, axes=([0, 1], [0, 1]))
            C[n] = c[n] - m_op[n] * m_op[l]
            T = mps.extend_sidetensor(n, T, -1, mpo=False)

        c = c[::-1]
        C = C[::-1]
        mps.m = {**mps.m,
            'two-point': c,
            'correlation': C,
        }
        return

    def full_two_point(mps, **P):

        lm = min(int(mps.L*P['points']), mps.L//2)
        left = mps.L//2-lm//2+mps.L%2-1
        right = mps.L//2+lm//2

        c = np.zeros([mps.L, mps.L], dtype=mps.dtype)
        C = np.zeros([mps.L, mps.L], dtype=mps.dtype)

        m_op = mps.m['one-point']

        for l in range(right, left, -1):
            mps.move_border_to(l)

            T = mps.contract(l, mps.O, axes=-1)

            lim = max(l-1-P['max_distance'], left-1)
            for n in range(l-1, lim, -1):
                k = mps.contract(n, mps.O, axes=+1)
                c[l, n] = np.tensordot(k, T, axes=([0, 1], [0, 1]))
                C[l, n] = c[l, n] - m_op[l] * m_op[n]
                c[n, l] = c[l, n]
                C[n, l] = C[l, n]
                T = mps.extend_sidetensor(n, T, -1, mpo=False)

        mps.m = {**mps.m,
            'full two-point': c,
            'full correlation': C,
        }
        return

