
def Heisenberg(P):

    sc = []
    for name, val in P['Simulation']['couplings'].items():
        if name == 'J':
            st = ['{:.2f}'.format(eval(v)) for v in val]
            sc.append('${}=({})$'.format(name, ','.join(st)))
        else:
            sc.append('${}={:.2f}$'.format(name, eval(val)))

    return f'{header(P)} {" ".join(sc)}'

def Rydberg(P):

    sc = []
    for name, val in P['Simulation']['couplings'].items():
        if name != 'V':
            sc.append('${}={:.3f}$'.format(name, eval(val)))

    return f'{header(P)} {" ".join(sc)}'

############################

def header(P):
    return '{} {}: $L={}$ $D={}$'.format(
        P['Data']['tags'][0],
        P['System'],
        P['Simulation']['L'],
        P['Simulation']['svd']['cutoff'],
        )

def couplings(P):
    sc = []
    for name, val in P['Simulation']['couplings'].items():
        if type(val) is list:
            val = ['{:.2e}'.format(eval(i)) for i in val]
            sc.append('${}=({})$'.format(name, ' '.join(val)))
        else:
            sc.append('${}={:.2e}$'.format(name, eval(val)))

    return ' '.join(sc)

def title(P):
    return globals().get(P['System'], lambda P: f'{header(P)} {couplings(P)}')(P)


