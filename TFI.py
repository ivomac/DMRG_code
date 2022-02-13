#!/usr/bin/env python3

L = 12

P = {
    'System': 'Heisenberg',
    'Simulation':
    {
        'L': L,
        'couplings': {
            'J': ['0', '0', '1'],
            'h_x': '0.5',
        },
        'svd': {
            'minimum_value': 0,
            'cutoff': 10000,
        },
        'dmrg': {
            'min_sweeps': 6,
            'precision': 1e-1,
        },
    },
    'Initialization': {
        'Cutoff': {
            'growth': 10000,
            'initial_not_same': 10000,
            'initial_same': 10000,
            'restarted': 10000,
        },
        'ED': {'use': False, 'L': 12},
        'random': {'L': 12},
    },
}

from sys import argv
from lib.start import start
start(P, argv)

