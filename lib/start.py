#!/usr/bin/env python3

from pathlib import Path
from lib.data import json_hash, insert_dic

NaN = float('nan')

def start(P, argin):

    from lib import default
    P = getattr(default, P['System'])(P)

    for fun in argin[1:]:
        Fun = globals()[fun]
        Fun(P)
    return

def Map(P, fun):

    def multi_range(grid_sweep):
        def recursive_range(it):
            if not it:
                yield ()
            else:
                for item in it[0]:
                    for rest_tuple in recursive_range(it[1:]):
                        yield (item,) + rest_tuple

        lengths = list(map(lambda x: len(x[1]), grid_sweep))
        return recursive_range(list(map(range, lengths)))

    for Po in P['Data']['subsets']:
        Po = insert_dic(P, Po)
        linear_range = Po['Data'].get('linear_range', None)
        grid = Po['Data'].get('grid', None)
        if linear_range is not None:
            if Po['Data']['only_edges']:
                for n in range(len(linear_range)):
                    linear_range[n][1] = [linear_range[n][1][0], linear_range[n][1][-1]]
            for i in range(len(linear_range[0][1])):
                for n in range(len(linear_range)):
                    Po['Simulation']['couplings'][linear_range[n][0]] = linear_range[n][1][i]
                fun(Po)
        elif grid is not None:
            if Po['Data']['only_edges']:
                for n in range(len(grid)):
                    grid[n][1] = [grid[n][1][0], grid[n][1][-1]]
            for comb in multi_range(grid):
                for n in range(len(comb)):
                    Po['Simulation']['couplings'][grid[n][0]] = grid[n][1][comb[n]]
                fun(Po)
        else:
            fun(Po)
    return

def ED(P):

    from lib import ED

    E_ED, psi = ED.state(
        getattr(ED, P['System'])(P['Simulation'], P['Initialization']['ED']),
        P['Simulation']['state']
    )

    print('Energy:', E_ED/P['Simulation']['L'])
    return E_ED, psi

def run(P):

    try:
        from lib.data import data
        DATA = data(P)
        DATA.run()
    except SystemExit:
        pass
    except:
        from traceback import format_exc

        message = '\nFailed\nError Message:\n' + format_exc()

        if 'DATA' in locals():
            message += '\n\n' + DATA.title.replace('$', '') + '\nReport:\n'
            if DATA.MPS is not None:
                message += '\n\nMPS Debug:\n'
                message += '\nmps shape:\n'
                for l in range(len(DATA.MPS)):
                    if DATA.MPS[l] is not None:
                        message += str(l) + ': ' + str(DATA.MPS[l].shape) + '\n'
                    else:
                        message += str(l) + ': None\n'
            DATA.rm('.running')
        else:
            from pathlib import Path
            from lib.data import json_hash
            fl = Path(P['Data']['folder']) / 'Data' / P['System'] / json_hash(P['Simulation']) / '.running'
            if fl.is_file():
                fl.unlink()

        print(message)

    return

def Run(P):
    Map(P, run)
    return

def Clean(P):

    def clean(P, files=[], keep_if=['MPS', 'temp_MPS', '.running']):
        df = Path(P['Data']['folder']) / 'Data' / P['System'] / json_hash(P['Simulation'])

        def delete(folder):
            for fl in folder.glob('*'):
                print(f'Deleting {fl.name}')
                fl.unlink()
            print(f'Deleting folder')
            folder.rmdir()
            return

        if df.is_dir():
            for fl in files:
                if (df / fl).is_file():
                    print(f'Deleting {fl}')
                    (df / fl).unlink()
            if all(map(lambda n: not (df / n).is_file(), keep_if)):
                delete(df)

    Map(P, clean)
    return

def Refit(P):

    from lib.data import data

    def refit(P):
        DATA = data(P)
        DATA.refit()
        return

    Map(P, refit)
    return

def Exponent(P):

    from pprint import pformat

    from lib.data import data

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    Subsets = P['Data']['subsets']

    quants = P['Exponent'].keys()

    M = []
    for n in range(len(Subsets)):
        Ps = insert_dic(P, Subsets[n])
        L = Ps['Simulation']['L']
        Range = Ps['Data']['linear_range']
        M.append({})

        for q in quants:
            M[n][q] = {'x': [[], []], 'C': [[], []], 'dx': [[], []]}

        for d in range(len(Range[0][1])):
            for j in range(len(Range)):
                Ps['Simulation']['couplings'][Range[j][0]] = Range[j][1][d]
            DATA = data(Ps)
            Consts = DATA.get('Constants')
            if Consts is not None:
                for q in quants:
                    p = Ps['Exponent'][q]
                    if p['plot']:
                        t, dt = Consts[q]
                        t = abs(t - p['shift'])
                        x = float(Range[0][1][d])
                        if (p['lims'][0] <= t <= p['lims'][1] and
                            p['x_lims'][0] <= x <= p['x_lims'][1] and x not in p['rm']):
                            i = (p['Fit']['lims'][0] <= t <= p['Fit']['lims'][1] and
                                p['Fit']['x_lims'][0] <= x <= p['Fit']['x_lims'][1] and x not in p['Fit']['rm'])
                            if p['x_shift'] is not None:
                                x = abs(x-p['x_shift'])
                            M[n][q]['x'][i].append(x)
                            M[n][q]['C'][i].append(1./t if p['invert'] else t)
                            M[n][q]['dx'][i].append(1./t**2*dt if p['invert'] else dt)

    plot = {}
    for q in quants:
        has_data = False
        for n in range(len(Subsets)):
            if len(M[n][q]['C'][0]) or len(M[n][q]['C'][1]):
                has_data = True
                break

        if has_data:

            Ps = insert_dic(P, Subsets[0])
            plot[q] = {}
            fs = plt.rcParams['figure.figsize']
            siz = Ps['Exponent_Properties']['size']
            plot[q]['fig'], plot[q]['ax'] = plt.subplots(figsize=(fs[0]*siz[0], fs[1]*siz[1]), constrained_layout=True)

            plot[q]['path'] = P['Data']['folder'] + f'/Plots/{q}.pdf'

            plot[q]['ax'].set_xlabel('${:s}$'.format(Ps['Data']['linear_range'][0][0]))
            plot[q]['ax'].set_ylabel(f'${Ps["Exponent"][q]["label"]}$')

            for n in range(len(Subsets)):
                plot[q][n] = {
                    'plot': None,
                    'fit_plot': None,
                    'fit_params': None,
                }

    for n in range(len(Subsets)):

        Ps = insert_dic(P, Subsets[n])
        L = Ps['Simulation']['L']
        Range = Ps['Data']['linear_range']

        for q in plot.keys():

            p = Ps['Exponent'][q]['Fit']

            x = M[n][q]['x'][1]
            C = M[n][q]['C'][1]
            dx = M[n][q]['dx'][1]

            if len(x) < 4:
                sen = 'Not enough data for exponent fit...'
                print(sen)
                continue

            def f(k, x):
                res = np.zeros(len(x), dtype=C[0].dtype)
                for i in range(len(x)):
                    v = x[i] >= k[0]
                    res[i] = k[2+v] * abs(x[i] - k[0])**k[1]
                return res

            mean = (max(x) + min(x))/2
            dif  = (max(x) - min(x))

            index_min = min(range(len(C)), key=C.__getitem__)

            lml = min(x)-p['extend']*dif
            lmh = max(x)+p['extend']*dif
            if p['crit_point'] is None:
                if p['x_start'] is not None:
                    crit_est = p['x_start']
                elif index_min in [0, len(C)-1]:
                    crit_est = x[index_min]
                else:
                    crit_est = mean
            else:
                crit_est = p['crit_point']
                lml, lmh = p['crit_point']-1e-15, p['crit_point']+1e-15

            from lib.fit import ls
            r = ls(Ps['Fit']['Least Squares'], lambda k: f(k, x) - C,
                    [crit_est, 1. if p['expected'] is None else p['expected'], C[0]/abs(mean), C[-1]/abs(mean)],
                    bounds=([lml, 0, 0, 0],
                            [lmh, 5., np.inf, np.inf]))

            rep_str = f'L={L}\n'
            rep_str += '{:s}={:.4f}'.format(p['exp'], r.x[1])
            if p['expected']:
                rep_str += '({:+.1f})%'.format((r.x[1]-p['expected'])/p['expected']*100)
            rep_str += '\n{:s}_c={:.8e}\n'.format(Range[0][0], r.x[0])
            print(rep_str)

            plot[q][n]['fit_params'] = r.x

            if p['plot']:
                props = Ps['Exponent_Properties']['fit'][Ps['id']]
                if P['Exponent'][q]['log']:
                    lims = [x[0], x[-1]]
                else:
                    lims = [x[0], x[-1], r.x[0]]
                z = np.linspace(min(lims), max(lims), num=2000)
                plot[q][n]['fit_plot'] = plot[q]['ax'].plot(z, f(r.x, z), **props)[0]


    for n in range(len(Subsets)):

        Ps = insert_dic(P, Subsets[n])

        for q in plot.keys():

            p = Ps['Exponent'][q]

            for i in [0, 1]:

                tp = 'fit_data' if i else 'data'
                props = Ps['Exponent_Properties'][tp][Ps['id']]

                x = M[n][q]['x'][i]
                C = M[n][q]['C'][i]
                dx = M[n][q]['dx'][i]

                if q == 'E':
                    x, C, dx = treat_E(x, C, dx)

                if len(x):

                    if props['elinewidth'] is not None:
                        for h in range(len(dx)):
                            if abs(dx[h]) < p['Error']['minimum']:
                                dx[h] = NaN
                        dx_ops = {}

                        plot[q][n]['plot'] = plot[q]['ax'].errorbar(x, C, yerr=dx, **props)[0]
                    else:
                        del props['elinewidth']
                        del props['capsize']
                        plot[q][n]['plot'] = plot[q]['ax'].plot(x, C, **props)[0]

            plot[q]['ax'].set_title(Ps.get('title', ''))

    for q in plot.keys():
        leg = []

        ylim = plot[q]['ax'].get_ylim()
        if P['Exponent'][q]['log']:
            from matplotlib import ticker
            plot[q]['ax'].set_xscale('log')
            plot[q]['ax'].set_yscale('log')
            plot[q]['ax'].xaxis.set_minor_formatter(ticker.ScalarFormatter())
            plot[q]['ax'].yaxis.set_minor_formatter(ticker.ScalarFormatter())
        else:
            ylim = [0, ylim[1]]
            plot[q]['ax'].set_ylim(ylim)

            loc = plot[q]['ax'].get_yticks()
            rng = np.arange(0, ylim[1], step=10**np.floor(np.log10(ylim[1])))
            if len(rng) > 6:
                rng = rng[::2]
            plot[q]['ax'].set_yticks(rng)

        for n in range(len(Subsets)):
            Ps = insert_dic(P, Subsets[n])
            p = Ps['Exponent'][q]

            props = Ps['Exponent_Properties']
            leg_props = {},
            if plot[q][n]['plot'] is not None:
                leg_props = {
                    'label': f'${Ps["id"]}$',
                    **props['fit_data'][Ps['id']],
                }

                if plot[q][n]['fit_plot'] is not None:
                    leg_props = {
                        **leg_props,
                        'label': f'${Ps["id"]}$: ${p["Fit"]["exp"]} = {plot[q][n]["fit_params"][1]:.2f}$',
                        **props['fit'][Ps['id']],
                    }

                del leg_props['elinewidth']
                del leg_props['capsize']
                leg.append(Line2D([0], [0], **leg_props))

                crit_label = p['add_crit_line']
                if not P['Exponent'][q]['log'] and plot.get(crit_label, None) is not None and plot[crit_label][n]['fit_params'] is not None:
                    crit_point = plot[crit_label][n]['fit_params'][0]
                    plot[q]['ax'].plot([crit_point, crit_point], ylim, **props['critical_point'][Ps['id']])

        plot[q]['ax'].legend(handles=leg, **P['Legend'])

        plot[q]['fig'].savefig(plot[q]['path'])
        plt.close(plot[q]['fig'])

    if len(Subsets) > 2 and plot.get('ξ', None) is not None and P['Exponent']['ξ']['Fit']['expected'] is not None:

        plot['scl'] = {}
        plot['scl']['fig'], plot['scl']['ax'] = plt.subplots(1, 1)

        plot['scl']['path'] = P['Data']['folder'] + f'/Plots/scaling.pdf'

        plot['scl']['ax'].set_xlabel('$L^{-1/\\nu}$')
        plot['scl']['ax'].set_ylabel('${:s}$'.format(P['Data']['linear_range'][0][0]))

        exp = P['Exponent']['ξ']['Fit']['expected']

        x = []
        y = []
        for n in range(len(Subsets)):
            Ps = insert_dic(P, Subsets[n])
            L = Ps['Simulation']['L']
            if L not in x:
                x.append(L**(-1/exp))
                y.append(plot[q][n]['fit_params'][0])

        r = np.polyfit(x, y, 1)
        rep_str = '\n{:s}_c={:.8e}\n'.format(Range[0][0], r[1])
        print(rep_str)

        z = np.linspace(0, max(x), num=10)
        plot['scl']['ax'].plot(z, [r[0]*i+r[1] for i in z])

        plot['scl']['ax'].plot(x, y, linestyle='', marker='.')

        plot['scl']['fig'].savefig(plot['scl']['path'])
        plt.close(plot['scl']['fig'])

    for n in range(len(Subsets)):
        Ps = insert_dic(P, Subsets[n])

    T = f'Exponent {P["Data"]["tags"][-1]}'

    return

def Scatter(P):

    from pprint import pformat

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from lib.data import data
    from lib.title import header

    Grid = P['Data']['grid']

    kw = P['Scatter']
    Ps = P['Scatter_Properties']

    quants = []

    for q, p in kw.items():
        if p['plot']:
            quants.append(q)
            if p['Contour']['plot']:
                quants.append(p['Contour']['quantity'])

    quants = set(quants)

    C = {}
    for q in quants:
        C[q] = [
            [ NaN for c in range(len(Grid[1][1])) ]
            for b in range(len(Grid[0][1]))
        ]

    x = list(map(float, Grid[0][1]))
    y = list(map(float, Grid[1][1]))

    X = np.array([ x for b in range(len(Grid[1][1])) ]).T
    Y = np.array([ y for b in range(len(Grid[0][1])) ])

    for d in range(len(Grid[0][1])):
        P['Simulation']['couplings'][Grid[0][0]] = Grid[0][1][d]
        for o in range(len(Grid[1][1])):
            P['Simulation']['couplings'][Grid[1][0]] = Grid[1][1][o]
            DATA = data(P)
            Consts = DATA.get('Constants')
            if Consts is not None:
                for q in quants:
                    C[q][d][o] = Consts[q][0]
                    if q == 'q' and Consts['p'][0] > 0:
                        C[q][d][o] = NaN

    figpaths = []
    line_txts = []

    for q, p in kw.items():
        if p['plot']:
            fs = plt.rcParams['figure.figsize']
            fig, ax = plt.subplots(figsize=(fs[0]*Ps['size'][0], fs[1]*Ps['size'][1]), constrained_layout=True)

            props_pcolormesh = Ps['Colormesh']
            props_colorbar = Ps['Colorbar']

            if q == 'p':
                phases = P['Phases']
                cbmin, cbmax = min(phases['indices']), max(phases['indices'])
                props_pcolormesh = {**props_pcolormesh,
                    'cmap': ListedColormap( phases['colors'] ),
                    'vmin': cbmin-0.5, 'vmax': cbmax+0.5
                }
                props_colorbar = {**props_colorbar, 'ticks': phases['indices']}

            psm = ax.pcolormesh(X, Y, C[q], **props_pcolormesh)
            if p['colorbar']:
                colbar = fig.colorbar(psm, ax=ax, **props_colorbar)
                colbar.ax.yaxis.set_offset_position('left')
                colbar.ax.set_title(f'${p["label"]}$')
                ax.set_title(f'${p["label"]}$')
                if q == 'p':
                    colbar.ax.set_yticklabels(phases['labels'])
                    colbar.ax.tick_params(right=False)

            if p['Contour']['plot']:
                props = p['Contour']
                levels = []
                for h in range(len(props['lims'][::3])):
                    levels += list(np.linspace(props['lims'][3*h+0], props['lims'][3*h+1], props['lims'][3*h+2]))
                levels.sort()
                dic, lvls = {}, []
                for n in props['add_lines']:
                    lvls.append(eval(n))
                    dic[eval(n)] = n
                    for k in range(len(levels)-1,-1,-1):
                        if abs(eval(n) - levels[k]) < 1e-13:
                            levels.pop(k)
                for n in props['rm_lines']:
                    for k in range(len(levels)-1,-1,-1):
                        if abs(eval(n) - levels[k]) < 1e-13:
                            levels.pop(k)

                psm = ax.contour(X, Y, np.array(C[props['quantity']]), levels, **Ps['Contour'], zorder=9)
                ax.clabel(psm, fmt=(lambda n: (f'{{:.{props["accuracy"]}f}}').format(n)), **Ps['Contour_label'])

                if props['add_lines']:
                    psm = ax.contour(X, Y, np.array(C[props['quantity']]), lvls, **Ps['Contour_special'])
                    ax.clabel(psm, fmt=dic, **Ps['Contour_label'])

                    curves = psm.collections
                    for k in range(len(props['add_lines'])):
                        curve_x, curve_y = [], []
                        curve = curves[k].get_segments()
                        for segment in curve:
                            curve_x += list(segment.T[0])
                            curve_y += list(segment.T[1])
                        if len(curve_x):
                            JJ = np.polyfit(curve_x, curve_y, 1)
                            line_txts.append(f'{props["add_lines"][k]}: {list(JJ)}')

            for line in p['lines']:
                x = [eval(line[2]), eval(line[3])]
                m, b = line[1]
                y = [m*xi + b for xi in x]
                if line[0][0] == Grid[0][0]:
                    ax.plot(x, y, **Ps['Lines'])
                    ax.text(x[-1]+line[4], y[-1]+line[5], line[6])
                else:
                    ax.plot(y, x, **Ps['Lines'])
                    ax.text(y[-1]+line[5], x[-1]+line[4], line[6])
            for point in p['points']:
                ax.plot(point[0], point[1], **Ps['Points'])

            ax.set_xlabel(f'${Grid[0][0]}$')
            ax.set_ylabel(f'${Grid[1][0]}$')

            from matplotlib.ticker import FormatStrFormatter
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            figpaths.append( P['Data']['folder'] + f'/Plots/Phase_diagram_{q}.pdf' )

            fig.savefig(figpaths[-1])
            plt.close(fig)

    return


