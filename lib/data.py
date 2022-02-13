
from sys import exit
from datetime import datetime
from pathlib import Path
from pprint import pprint, pformat
import matplotlib.pyplot as plt
from math import sqrt

from copy import copy

from pickle import loads, dumps


def json_hash(P):
    from hashlib import sha256
    from json import dumps as Jdumps
    return sha256(Jdumps(P, sort_keys=True).encode('utf8')).hexdigest()


def insert_dic(base_dic, dic):
    new_dic = copy(base_dic)
    for key in dic.keys():
        if type(dic[key]) == dict and type(new_dic.get(key, None)) == dict:
            new_dic[key] = insert_dic(new_dic[key], dic[key])
        else:
            new_dic[key] = dic[key]
    return new_dic


class dots:
    def __init__(self, *dtype, **kwargs):
        pass


class data:

    def __init__(data, Parameters):
        from os import getenv
        from lib import title

        data.P = Parameters
        for k in ['Data', 'Run', 'System', 'Simulation', 'Initialization', 'Measure']:
            setattr(data, k, Parameters[k])

        data.process_couplings()

        data.title = getattr(title, data.System, title.title)(data.P)
        data.report = ''

        fl = Path(data.Data['folder'])
        if fl.is_file():
            fl = fl.parent

        data.root = fl
        data.path = data.root / 'Data' / data.System
        data.folder = data.path / json_hash(data.Simulation)

        data.FIT = None
        data.MPS = None

        data.restarted = False

        data.Initial = {
            'hash': None,
            'parameters': None,
            'distance': None,
            'length_difference': None,
            'precision': None,
            'same': None,
            'replaced': None,
        }

        print(data.folder.name, '\n')
        print(pformat(data.Simulation), '\n')

        data.reply = ''
        data.start_time = datetime.now()

    @property
    def Constants(data):
        if data.FIT is not None:
            return data.FIT.Constants
        return

    @property
    def Parameters(data):
        return data.Simulation

    @property
    def Measurements(data):
        if data.MPS is not None:
            return data.MPS.m
        return

    def report_initial(data):
        if data.Initial['hash'] is not None:
            print('Initial:', pformat(data.Initial), '\n')

    def process_couplings(data):
        from lib.decimal import format_coupling
        for coup, value in data.Simulation['couplings'].items():
            param2 = data.Simulation['couplings'][coup]
            if type(value) is list:
                for i, val in enumerate(value):
                    data.Simulation['couplings'][coup][i] = format_coupling(val)
            else:
                data.Simulation['couplings'][coup] = format_coupling(value)
        return

    def param_diff(data, params):
        diff = 0.
        for coup, value in data.Simulation['couplings'].items():
            param2 = params['couplings'][coup]
            if type(value) is list:
                for i, val in enumerate(value):
                    diff += ( eval(val) - eval(param2[i]) )**2
            else:
                value = eval(value)
                param2 = eval(param2)
                diff += ( value - param2 )**2
        return sqrt(diff)

    def check(data, tp, mess=False):
        fl = data.folder / tp
        exist = fl.is_file()
        if exist and mess:
            print(f'{tp} found!', '\n')
        return exist

    def tag(data, *tags):

        def make_tag(t):
            tag_file = data.folder / t
            if tag_file.is_file():
                tag_file.unlink()
            tag_file.write_text('')
            return

        data.folder.mkdir(parents=True, exist_ok=True)

        for t in data.Data['tags']:
            make_tag('tag.' + t)

        for t in tags:
            make_tag(t)

        return

    def get(data, tp):
        fl = data.folder / tp
        if fl.is_file():
            return loads(fl.read_bytes())
        return

    def save(data, tp, temp=False):
        data.folder.mkdir(parents=True, exist_ok=True)

        prefix = 'temp_' if temp else ''
        pickle_file = data.folder / (prefix + tp)
        if pickle_file.is_file():
            pickle_file.unlink()
        pickle_file.write_bytes(dumps(getattr(data, tp), protocol=4))

        if not temp:
            temp_file = data.folder / ('temp_' + tp)
            if temp_file.is_file():
                temp_file.unlink()
        return

    def rm(data, tp):
        rm_file = data.folder / tp
        if rm_file.is_file():
            rm_file.unlink()
        return

    def init(data):
        if data.check('.running'):
            print(f'Simulation already running... closing.')
            exit()

        if data.Initialization['use_temp_mps'] and (data.folder / 'temp_MPS').is_file():
            print(f'Simulation resuming from stored MPS...')
            data.Initial = data.get('Initial')
            data.MPS = loads((data.folder / 'temp_MPS').read_bytes())
            data.MPS.set_params(data.P, full=False)
            data.restarted = True
            return

        if data.Initialization['use_start_mps']:

            for df in data.path.glob('*'):
                if (df / 'MPS').is_file():
                    try:
                        params = loads((df / 'Parameters').read_bytes())
                    except FileNotFoundError:
                        print(f'FileNotFound: Parameters - {df.name}', '\n')
                        continue

                    new = {
                        'hash': df.name,
                        'parameters': params,
                        'distance': data.param_diff(params),
                        'length_difference': abs(params['L'] - data.Simulation['L']),
                        'precision': 1/params['dmrg']['precision'],
                    }
                    if (new['distance']          <= data.Initialization['max_distance'] and
                        new['length_difference'] <= data.Initialization['max_length_difference']):
                        if data.Initial['hash'] is None:
                            data.Initial = new
                            continue
                        for j in ('distance', 'length_difference', 'precision'):
                            if data.Initial[j] > new[j]:
                                data.Initial = new
                                break
                            elif data.Initial[j] < new[j]:
                                break

            try:
                if data.Initial['hash'] is not None:
                    data.MPS = loads((data.path / data.Initial['hash'] / 'MPS').read_bytes())
                    data.MPS.set_params(data.P)
                    data.Initial['same'] = (
                        data.Initial['distance'] <= 1e-13 and
                        data.Initial['length_difference'] == 0
                    )
                    data.Initial['replaced'] = False
                    del data.Initial['precision']
                    data.save('Initial')
                    return
            except FileNotFoundError:
                print(f'FileNotFound: MPS - {data.Initial["hash"]}', '\n')

            print(f'Starting: No MPS found for starting point!', '\n')
            if data.Initialization['require_start_mps']:
                print(f'Starting MPS required... closing.', '\n')
                exit()

        from lib.mps import MPS
        data.MPS = MPS(data.P)
        return

    def message(data, time=False, foot=''):
        m = ''
        m = 'Sweep: {:d}\n'.format(data.MPS.sweeps)
        m += 'Energy density: {:.8e} ± {:.8e}\n'.format(
                data.MPS.m['energy density'][-1], data.MPS.m['energy density error'][-1])
        m += 'Relative error: {:.8e}\n'.format(
                data.MPS.m['relative energy density error'][-1])
        m += 'Max. bond dimension: {:d}\n'.format(data.MPS.max_D)
        m += 'Min. singular value: {:.2e}\n'.format(data.MPS.m.get('minimum singular value',[-1])[-1])
        if time:
            m += 'Running time: {!s}\n'.format(
                    datetime.now() - data.start_time)
        m += '\n' + foot + '\n'
        print(m, '\n')
        return

    def calculate_constants(data, title=False):
        from lib import fit

        fs = plt.rcParams['figure.figsize']
        fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(2*fs[0], 2*fs[1]*(1+0.1*title)))
        if title:
            fig.suptitle(data.title)

        Measurements = data.get('Measurements')
        data.FIT = fit.fit(data.P, Measurements)

        data.FIT.energy()
        data.FIT.scaling_dimension(ax1[0])
        data.FIT.central_charge(ax1[1])
        data.FIT.correlation_length(ax2)
        data.FIT.phase()

        data.save_figure(fig)
        print(pformat(data.FIT.Constants), '\n')
        return

    def save_figure(data, fig, filename=None):
        (data.root / 'Plots').mkdir(parents=True, exist_ok=True)
        if not filename:
            filename = 'Plot'
        for folder in [data.folder, data.root / 'Plots']:
            fl = folder / filename
            if not fl.suffix:
                fl = fl.with_suffix('.pdf')
            fig.savefig(fl)
        plt.close(fig)
        return

    def measure(data):
        for tp in ['one_point', 'nn_two_point', 'two_point', 'full_two_point']:
            v = data.Measure[tp]
            if v['measure']:
                getattr(data.MPS, tp)(**v)
        data.save('Measurements')
        return

    def remeasure(data):
        if data.check('MPS'):
            data.MPS = data.get('MPS')
            data.message()
            data.measure()
            data.refit()
        return

    def refit(data):
        if data.check('Measurements'):
            data.calculate_constants()
            data.save('Constants')
        return

    def run(data):

        if data.Run['remeasure']:
            data.remeasure()
            return
        if data.Run['refit']:
            data.refit()
            return

        if data.check('MPS', mess=True):
            data.tag('.running')
            if not (data.folder / 'Parameters').is_file():
                data.save('Parameters')
            data.MPS = data.get('MPS')
            data.Initial = data.get('Initial')
            data.report_initial()
            data.message()
        else:
            data.save('Parameters')
            data.init()
            data.tag('.running')
            data.save('Initial')
            data.report_initial()
            data.MPS.resize(data.Initial['same'], data.restarted)
            while not data.MPS.converged(data.Initial['same'] and data.Run['stop_if_initial_same']):
                data.message(time=True)
                max_sweeps = data.MPS.sweeps >= data.Run['max_sweeps']
                max_bond = data.MPS.max_D == data.Simulation['svd']['cutoff'] and data.Run['stop_at_max_D']
                if max_sweeps or max_bond:
                    err_txt = 'sweeps' if max_sweeps else 'bond dimension'
                    print(f'Max {err_txt} reached!\n')
                    data.rm('.running')
                    exit()
                data.MPS.dmrg_sweep()
                data.save('MPS', temp=True)
            data.save('MPS')
            data.message(time=True, foot='Convergence reached!')

        if data.Initialization['replace'] and data.Initial['same']:
            initial_folder = data.path / data.Initial['hash']
            if initial_folder.is_dir():
                for fl in initial_folder.glob('*'):
                    fl.unlink()
                initial_folder.rmdir()
                data.Initial['replaced'] = True
                data.save('Initial')
                print('Initial MPS removed!', '\n')

        if not data.check('Measurements'):
            data.measure()

        if not data.check('Constants'):
            data.calculate_constants()
            data.save('Constants')

        data.rm('.running')
        return

