
import numpy as np
from math import pi as π

NaN = float('nan')

def ls(P, *args, **kwargs):
    from scipy.optimize import least_squares
    return least_squares(*args,
            max_nfev=P['max_iterations'],
            gtol=P['precision'],
            xtol=P['precision'],
            ftol=P['precision'],
            **kwargs)


def Heisenberg(M, P, C, phase_info):

    S = M['one-point']

    L = len(S)
    even = sum(S[::2])
    odd  = sum(S[1::2])

    O_F = [(even+odd)/L, NaN]
    O_AF = [(even-odd)/L, NaN]
    O = [abs(O_F[0]) - abs(O_AF[0]), NaN]

    if O[0] > abs(P['threshold']):
        p = 1
    elif O[0] < -abs(P['threshold']):
        p = -1
    else:
        p = 0

    phase_label = phase_info['labels'][phase_info['indices'].index(p)]

    Constants = {
        'O_F': O_F,
        'O_AF': O_AF,
        'O': O,
        'p': [p, phase_label],
    }

    return Constants


def Rydberg(M, P, C, phase_info):

    N = M['one-point']

    k = max(P['periods'])

    L = len(N)

    l = L//2-k//2-2
    r = L//2+k//2+3

    argmx = N[l:r].argmax()+l
    argmn = N[l:r].argmin()+l
    O = N[argmx]-N[argmn]

    mean = np.mean(N[l:r])

    n = np.zeros(L, dtype=int)
    for i in range(len(n)):
        n[i] = 1 if N[i]-mean >= 0 else 0

    res = np.zeros(len(P['periods']), dtype=n.dtype)
    for i in range(len(res)):
        ref = np.zeros(L, dtype=int)
        ref[::P['periods'][i]] = 1

        res[i] = sum(abs(n-ref))

    err = NaN
    mn = min(res)/L
    if (O  >= P['Threshold']['order_parameter'] and
        mn <= P['Threshold']['occupation']):
        p = P['periods'][res.argmin()]
        err = mn
    elif C['c'][0] >= P['Threshold']['min_c'] and C['c'][0] * mn/L >= P['Threshold']['floating']:
        p = -1
    elif abs(C['q'][0] - 1/2) <= P['Threshold']['commensurate']:
        p = 1
    else:
        p = 0

    phase_label = phase_info['labels'][phase_info['indices'].index(p)]

    if P['expected_period'] is None:
        R, R_err = NaN, NaN
    else:
        dq = abs(C['q'][0] - 1/P['expected_period'])
        R = dq*C['ξ'][0]
        R_err = dq*C['ξ'][1] + C['q'][1]*C['ξ'][0]

    return {'O': [O, err], 'p': [p, phase_label], 'R': [R, R_err]}


class fit:
    def __init__(fit, P, Measurements):

        fit.phase_info = P['Phases']

        fit.P = P['Fit']
        fit.O = eval(P['System'])
        fit.Measurements = Measurements
        fit.Constants = {}

    def ls(fit, *args, **kwargs):
        return ls(fit.P['Least Squares'], *args, **kwargs)

    def energy(fit):
        fit.Constants.update(
            {'E': (fit.Measurements['energy density'][-1], fit.Measurements['energy density error'][-1])}
        )
        return

    def linearize(fit, x, C, remove_edges=True):
        C_lin = list(C)
        x_lin = list(x)
        a = True
        while a:
            a = False
            for i in range(len(C_lin)-2, 0, -1):
                if len(C_lin) <= fit.P['linearize']:
                    a = False
                    break
                if C_lin[i-1] > C_lin[i] < C_lin[i+1]:
                    del C_lin[i]
                    del x_lin[i]
                    a = True

        if remove_edges:
            while len(C_lin) >= 3 and C_lin[0] < C_lin[1]:
                del C_lin[0]
                del x_lin[0]

            if len(C_lin) >= 3:
                del C_lin[-1]
                del x_lin[-1]

        C_lin = np.array(C_lin, dtype=C.dtype)
        x_lin = np.array(x_lin, dtype=C.dtype)
        return x_lin, C_lin

    def phase(fit):
        fit.Constants.update(
            fit.O(
                fit.Measurements,
                fit.P['Phase'],
                fit.Constants,
                fit.phase_info,
            )
        )
        return

    def scaling_dimension(fit, ax):

        def get_d(x, n):
            r, Cov = np.polyfit(np.log(x), np.log(n), 1, cov=True)
            err = np.sqrt(np.diag(Cov))
            d = [-r[0], err[0]]
            a = [r[1], err[1]]
            return d, a

        P = fit.P['Scaling Dimension']
        no = fit.Measurements['one-point']

        if not P['fit']:
            L = len(no)
            l = np.array(range(1, L+1))
            if P['plot']:
                ax.plot(l, no, marker='.', linestyle='', label='$\\langle \\hat{n}_{l} \\rangle$')
                ax.legend()
                ax.set_xlabel('$l$')
            fit.Constants.update(
                {
                    'd': [NaN, NaN],
                }
            )
            return

        period = fit.P['Phase']['expected_period']
        period = 2 if period is None else period

        no = fit.Measurements['one-point']

        L = len(no)
        l = np.array(range(1, L//2+1, period))

        n = np.array([abs(no[i-1] - min(no[i:i-1+period])) for i in l])

        x = np.sin(π*l/L)
        p = L

        x_l, n_l = fit.linearize(x, n, remove_edges=False)

        lm = [P['x_lims'][0] <= x_l[i] <= P['x_lims'][1] for i in range(len(x_l))]

        lm2 = [P['x_lims'][0]*0.98 <= x[i] for i in range(len(x))]

        x_l = x_l[lm]
        n_l = n_l[lm]

        shift = eval(P['shift'])
        label = f'{P["shift"]}' if shift else ''

        n_l = abs(n_l + shift)

        if len(x_l) < 4:
            d = [NaN, NaN]
        else:
            d, a = get_d(p*x_l, n_l)

            if P['plot']:
                ax.loglog(x[lm2], abs(n[lm2]+shift), marker='.', linestyle='', label='$\\langle \\hat{n}_l - \\hat{n}_{l-1} \\rangle$')
                ax.loglog(x_l, n_l, marker='.', linestyle='', color='tab:green')
                ax.loglog(x_l, np.exp(a[0])*(p*x_l)**(-d[0]), marker='', linestyle='-', linewidth=1, label='$-d \\log(L x_l) + a$')
                ax.legend()
                ax.set_title('$d = {:.3f} \\pm {:.2e}$'.format(d[0], d[1]))
                ax.set_xlabel('$x_l=\\sin(\\pi l/L)$')

        fit.Constants.update({'d': d})
        return

    def central_charge(fit, ax):

        def get_c(d, S):
            r, Cov = np.polyfit(np.log(d), S, 1, cov=True)
            c, u = 6*r[0], r[1]
            err = np.sqrt(np.diag(Cov))
            c = [c, 6*err[0]]
            u = [u, err[1]]
            return c, u

        P = fit.P['Central Charge']

        if not P['fit']:
            fit.Constants.update(
                {
                    'c': [NaN, NaN],
                    'u': [NaN, NaN],
                }
            )
            return

        S = fit.Measurements['entanglement entropy']

        L = len(S)+1

        S = S[:L//2]

        l = np.arange(0, L//2)
        x = np.sin(π*(l+1)/L)
        p = 2*L/π

        u = [-1, 0]

        x_l, S_l = fit.linearize(x, S, remove_edges=False)

        lm = [P['x_lims'][0] <= x_l[i] <= P['x_lims'][1] for i in range(len(x_l))]

        lm2 = [P['x_lims'][0]*0.99 <= x[i] for i in range(len(x))]

        x_l = x_l[lm]
        S_l = S_l[lm]

        if len(S_l) < 3:
            c = [NaN, NaN]
            u = [NaN, NaN]
        else:
            c, u = get_c(p*x_l, S_l)

            if c[0] < 0:
                c = [NaN, NaN]
                u = [NaN, NaN]

        if P['plot']:
            ax.semilogx(x[lm2], S[lm2], marker='.', linestyle='', color='tab:blue', label='$S_l$')
            ax.semilogx(x_l, S_l, marker='.', linestyle='', color='tab:green')
            if c[0] == c[0]:
                ax.semilogx(x_l, c[0]/6*np.log(p*x_l)+u[0], marker='', linestyle='-', linewidth=1, color='tab:orange', label='$\\frac{c}{6}\\log\\left(\\frac{2L}{π}x_l\\right)+u$')
            ax.set_title('$c = {:.3f} \\pm {:.2e}$'.format(c[0], c[1]))
            ax.set_xlabel('$x_l=\\sin(\\pi l/L)$')
            ax.legend()

        fit.Constants.update({'c': c, 'u': u})
        return


    def correlation_length(fit, ax):

        P = fit.P['Correlation Length']

        if not P['fit']:
            fit.Constants.update(
                {
                    'ξ': [NaN, NaN],
                    'A': [NaN, NaN],
                    'q': [NaN, NaN],
                }
            )
            return

        C = np.array(fit.Measurements['correlation'])
        L = len(C) + 1
        x = np.array(range(1, L))

        for i in range(len(x)-1, 0, -1):
            if C[i] > P['error_threshold']:
                P['x_lims'][1] = min(P['x_lims'][1], x[i]/L)
                break

        lm = [P['x_lims'][0] <= x[i]/L <= P['x_lims'][1] for i in range(len(x))]

        def get_corr(x, C):
            x_l, C_l = fit.linearize(x, np.log(abs(C)*x**P['exponent']))

            lm = [P['x_lims'][0] <= x_l[i]/L <= P['x_lims'][1] for i in range(len(x_l))]
            x_l = x_l[lm]
            C_l = C_l[lm]

            if len(C_l) < 3:
                ξ = [NaN, NaN]
                A = [NaN, NaN]
            else:
                r, Cov = np.polyfit(x_l, C_l, 1, cov=True)
                ξ, A = -1/r[0], np.exp(r[1])
                err = np.sqrt(np.diag(Cov))
                ξ = [ξ, err[0]*ξ**2]
                A = [A, err[1]*A]
            return ξ, A

        def div(x, C, p):
            return C*x**P['exponent']*np.exp(x/p['ξ'][0])/p['A'][0]

        def f(k, x):
            return k[1]*np.cos(2*π*k[0]*x)

        def dq(x, C, q, A):
            c = sum( (f([q, A], x)-C)**2 )
            dc = sum(abs(
                (f([q, A], x)-C)*A*np.cos(2*π*q*x)*2*π*x
            ))
            return c/dc

        def sig(x, C, q):
            mx = abs(C).argmax()
            return np.sign(C[mx]*f([q, 1], x[mx]))*abs(C[mx])

        def estimate_q(x, C):
            q = np.linspace(P['q_lims'][0], P['q_lims'][1], num=P['q_points'], dtype=C.dtype)
            kj = np.abs(np.tensordot(
                np.cos(2*π*np.tensordot(q, x, axes=0)),
                C,
            axes=1))

            q = q[np.argmax(kj)]

            Al = sig(x, C, q)

            return [q, dq(x, C/Al, q, 1)]

        def get_q(x, C):

            def jac(k, x):
                J = np.zeros((len(x), 2), dtype=C.dtype)
                J[:, 0] = -x*2*π * k[1] * np.sin(2*π*k[0]*x)
                J[:, 1] = 1             * np.cos(2*π*k[0]*x)
                return J

            q = estimate_q(x, C)

            Al = sig(x, C, q[0])

            lm = np.sort([0.5*Al, 2*Al])

            r = fit.ls(lambda k, x: f(k, x) - C,
                [q[0], Al], args=[x], jac=jac,
                bounds=([P['q_lims'][0], lm[0]],
                        [P['q_lims'][1], lm[1]])
            )

            return [r.x[0], dq(x, C/r.x[1], r.x[0], 1)], [r.x[1], NaN]

        def full_fit(x, Co, p):

            C = np.log(abs(Co)*x**P['exponent'])

            def F(k, x):
                return -x/k[0] + np.log(k[1]) + 0.5*np.log(np.cos(2*π*k[2]*x)**2)

            def jac(k, x):
                J = np.zeros((len(x), 3), dtype=C.dtype)
                J[:, 0] = x/k[0]**2
                J[:, 1] = 1/k[1]
                J[:, 2] = 2*π*x*np.tan(2*π*k[2]*x)
                return J

            Al = abs(p['A'][0])

            r = fit.ls(lambda k, x: F(k, x) - C,
                [p['ξ'][0], Al, p['q'][0]], args=[x], jac=jac,
                bounds=([1,        0.5*Al, P['q_lims'][0]],
                        [np.infty, 2*Al,   P['q_lims'][1]]),
            )

            k = {
                'ξ': [r.x[0], 10],
                'A': [r.x[1], NaN],
                'q': [r.x[2], 10],
            }

            Cd = div(x, Co, k)
            k['q'][1] = dq(x, Cd, k['q'][0], np.sign(sig(x, Co, k['q'][0])))

            return k['ξ'], k['A'], k['q']

        def normalize(x, p, l):
            mp = abs(np.cos(2*π*p['q'][0]*x))
            ind = mp <= l
            mp[ind] = l
            return mp

        def mess(p, suffix=''):
            if P['print']:
                print(p['ξ'][0], p['q'], suffix)

        p = {
            'ξ': [NaN, NaN],
            'A': [NaN, NaN],
            'q': [NaN, NaN],
        }
        r = dict(**p)

        p['ξ'], p['A'] = get_corr(x, C)

        if p['ξ'][0] == p['ξ'][0]:

            if P['q_start'] == 'auto':
                p['q'] = estimate_q(x[lm], div(x[lm], C[lm], p))
            else:
                p['q'] = [P['q_start'], NaN]

            mess(p)

            for cut in P['repeats']:

                r['ξ'], r['A'] = get_corr(x, C/normalize(x, p, cut))

                if r['ξ'][0] != r['ξ'][0]:
                    break

                r['q'], A = get_q(x[lm], div(x[lm], C[lm], r))
                r['A'] = [r['A'][0]*A[0], r['A'][1]*A[0]]

                if r['q'][1] > p['q'][1] and not P['force']:
                    break

                p = dict(**r)
                mess(p)

            if P['full_fit']:
                p['ξ'], p['A'], p['q'] = full_fit(x[lm], C[lm], p)
                mess(p, suffix='ff')

        if P['print']:
            print('')

        if P['plot']:
            ax[0].semilogy(x, abs(C), marker='.', color='tab:blue', linestyle='', label='$C_l$')
            if p['ξ'][0] == p['ξ'][0]:
                ax[0].semilogy(x[lm], p['A'][0]*np.exp(-x[lm]/p['ξ'][0])/x[lm]**P['exponent'], marker='', color='tab:orange', linestyle='-', linewidth=1, label='$A_l$')
            ax[0].legend()
            ax[0].set_title('$\\xi = {:.2f} \\pm {:.2e}$'.format(p['ξ'][0], p['ξ'][1]))
            ax[0].set_xlabel('$l$')
            ax[0].ticklabel_format(axis='x', style='plain')

            if p['ξ'][0] == p['ξ'][0]:
                ax[1].plot(x[lm], div(x[lm], C[lm], p), marker='.', color='tab:blue', linestyle='', label='$C_l/A_l$')
                ax[1].plot(x[lm], f([p['q'][0], 1], x[lm]), marker='+', color='tab:orange', linestyle='', label='$\\cos{(2\\pi ql+\\phi_0)}$')
                ax[1].legend()
                ax[1].set_title('$q = {:.4f} \\pm {:.2e}$'.format(p['q'][0], p['q'][1]))
                ax[1].set_xlabel('$l$')
                ax[1].ticklabel_format(axis='x', style='plain')

        fit.Constants.update(p)
        return

