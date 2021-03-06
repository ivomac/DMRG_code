# DMRG

(WIP, more documentation will be added)
This is a Python implementation of the DMRG algorithm in Matrix Product State form [[1]](https://arxiv.org/abs/1008.3477).

<!-- # Structure of code -->

# How to use

Modify and run the already provided scripts Rydberg.py and TFI.py (Transverse field Ising).

# How to add a model

To do DMRG on another model (let's call it "MODEL"), follow these steps:

1. Add a function to lib/default.py named "MODEL" returning a
   dictionary of parameters (example):

```
Parameters = {
  'System': 'MODEL',
  'd': 2,                 # Physical bond dimension
  'Simulation': {
    'L': L,
    'state': 0,           # State to be determined (0 for ground state)
    'dtype': 'float64',   # Change to complex64/128 for
                          # complex-valued MPS/MPO if required.
    'couplings': {
      'L': ['0', '1.05'], # Example couplings of Hamiltonian.
      'Z': '0.333',       # NOTE: the values are defined as strings!
                          # This is important.
    },
    'svd': {
      'minimum_value': 1e-8, # Singular values smaller than this are removed.
      'cutoff': 200,         # Maximum MPS virtual bond dimension.
    },
    'dmrg': {
      'min_sweeps': 4,    # Minimum #sweeps before simul. can stop.
      'precision': 1e-6,  # Simulation stops if
                          # sqrt(relative energy variance) < precision.
    },
  },
  'Fit': {
    'Phase': {            # These parameters are passed to the
      'threshold': 0.3,   # lib/fit.py 'MODEL' function (below).
    },                    # Define any items you wish.
  },
  'Phases': {
    'indices': (0, 1),
    'labels':  ('VBS', 'QSL'),
    'colors':  ('black', 'tab:gray'),
  },
}
```

Note that the 'System' key should match the name of all functions below.

2. Add a function to lib/mpo.py named "MODEL" returning as a numpy.array:
  - The bulk tensor M[i, s, s', j]:
    * i,j are the virtual indices and s,s' physical. The matrix (M[s, s'])[i, j] should be upper triangular. (See the example models already implemented).
  - A local operator:
	* to calculate its local expectation value and correlation function.

3. Add a function to lib/fit.py named "MODEL" returning a dictionary containing constants of interest (such as an order parameter). Example:
```
Constants = {
  'O': [f(M['one-point']), NaN],  # Order parameter
  'p': [0, 'VBS'],                # Phase
  'k': [??[0]**2, 2*??[0]*??[1]]     # Return whatever you wish
},
```

The second entries of (most) constants define the error/confidence interval (use 'NaN' if error is not available).

4. (Optional) Add a function to lib/ED.py named "MODEL" returning as a sparce
   matrix (csr_matrix) the open-boundary Hamiltonian for a general chain length. Otherwise,
   a random MPS will be used as starting point of the simulation.

5. (Optional) Add a function to lib/title.py named "MODEL" returning a string to
   be printed as title/header when the program is running.

6. Finally, create a run script in the main folder to run the code (use TFI.py and
   Rydberg.py as examples).

