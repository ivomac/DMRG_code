
from decimal import Decimal as D, getcontext


def format_coupling(val, precision=None):
    if precision is not None:
        getcontext().prec = precision
    return str( D(val).normalize() )


def decimal_range(start, stop, step):
    r = [start]
    while D(r[-1]) < D(stop):
        r.append( format_coupling( D(r[-1]) + D(step) ) )
    return r


def decimal_logspace(start, stop, step, base='10', precision=8):
    getcontext().prec = precision
    r = [format_coupling( D(base)**D(start) )]
    while D(start) < D(stop):
        start = format_coupling( D(start) + D(step) )
        r.append( format_coupling( D(base)**D(start) ) )
    return r


def decimals_in_line(r, x0, labels=('x', 'y'), precision=8):
    getcontext().prec = precision

    y0 = []
    for i in range(len(x0)):
        z = r[0]*float(x0[i]) + r[1]
        y0.append( format_coupling( z ) )
    return [[labels[0], x0], [labels[1], y0]]

def Range(labels, line, *lims):
    dr = []
    for l in lims:
        dr += decimal_range(l[0], l[1], l[2])
    return decimals_in_line(line,
    dr, labels=labels, precision=8)

