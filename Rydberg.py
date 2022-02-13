#!/usr/bin/env python3

from lib.decimal import decimal_range

l = 10
L = 12*l+1

Range = [
    ('x', decimal_range('1.82', '1.92', '0.02')[::1]),
]

Parameters = {
    'System': 'Rydberg',
    'Simulation':
    {
        'L': L,
        'couplings': {
            # 'x': '1.85',
            'y': '2.2',
            'V': '1',
        },
        'svd': {
            'minimum_value': 1e-8,
            'cutoff': 200,
        },
        'dmrg': {
            'min_sweeps': 4,
            'precision': 1e-6,
        },
    },
    'Fit': {
        'Phase': {
            'expected_period': 3,
        },
        'Correlation Length': {
            'q_lims': [0.3, 0.36],
        },
    },
    'Initialization': {
        'max_length_difference': 30,
        'max_distance': 0.05,
        'Cutoff': {
            'growth': 40,
            'initial_not_same': 100,
            'initial_same': 200,
            'restarted': 200,
        },
        'ED': {'use': True, 'L': 13},
        'random': {'L': 4},
    },
    'Data': {
        'linear_range': Range,
    },
    'Exponent': {
        'ξ': {
            'plot': True,
            'Fit': {
                'plot': True,
                'x_lims': [1.82, 1.92],
                'lims': [0, 35],
                'extend': 1,
            },
        },
        'q': {
            'plot': True,
            'Fit': {
                'plot': True,
                'x_lims': [1.82, 1.92],
            },
        },
    },
}

from sys import argv
from lib.start import start
start(Parameters, argv)

