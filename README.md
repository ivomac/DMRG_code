# DMRG

This is a Python implementation of the DMRG algorithm in Matrix Product State form [[1]](https://arxiv.org/abs/1008.3477).

# How to use

# Structure of code

# How to add a model

To do DMRG on another model (let's call it "MODEL"), follow these steps:

1. Add a function to lib/default.py named "MODEL" returning a
   dictionary of paramaters.

2. Add a function to lib/mpo.py named "MODEL" returning as a numpy.array:
  - the bulk tensor M[i, s, s', j]:
    * i,j are the virtual indices and s,s' physical. The matrix (M[s, s'])[i, j] should be upper triangular. (See the example models already implemented).
  - a local operator:
	* to calculate its local expectation value and correlation function.

3. (Optional) Add a function to lib/ED.py named "MODEL" returning as a sparce
   matrix (csr_matrix) the open-boundary Hamiltonian for a general chain length. Otherwise,
   a random MPS will be used as starting point of the simulation.

4. (Optional) Add a function to lib/title.py named "MODEL" returning a string to
   be printed as title/header when the program is running.


