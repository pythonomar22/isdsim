# Information Set Decoding (ISD) Simulator

A Python library for simulating and comparing classical Information Set Decoding (ISD) algorithms for decoding random linear codes.

## Overview

This library implements and provides tools to compare different classical Information Set Decoding (ISD) algorithms, which are used to solve the Syndrome Decoding Problem (SDP). The SDP is a fundamental problem in coding theory and forms the basis for many code-based cryptosystems.

The library includes:

- Core functionality for working with linear codes
- Implementation of classical ISD algorithms:
  - Prange's algorithm
  - Stern's algorithm
- Utilities for running experiments and comparing algorithm performance

## Installation

```bash
# Clone the repository
git clone https://github.com/pythonomar22/isd-lib.git
cd isd-lib

# Install using pip
pip install -e .
```

## Usage

### Basic Example

```python
import numpy as np
from isd_lib.core.linear_code import LinearCode
from isd_lib.algorithms.prange import PrangeISD
from isd_lib.algorithms.stern import SternISD

# Create a linear code
n, k = 20, 10
code = LinearCode(n, k)

# Create a random error vector of weight 2
e = np.zeros(n, dtype=int)
error_positions = np.random.choice(n, 2, replace=False)
e[error_positions] = 1

# Calculate the syndrome
s = code.syndrome(e)

# Decode with Prange's algorithm
prange = PrangeISD(max_iterations=1000)
e_prange, success_prange, iterations_prange = prange.decode(code.H, s, 2)

# Decode with Stern's algorithm
stern = SternISD(max_iterations=100, p=1, l=0)
e_stern, success_stern, iterations_stern = stern.decode(code.H, s, 2)
```

### Running Experiments

The library provides an `Experiment` class for running systematic experiments:

```python
from isd_lib.utils.experiment import Experiment

# Create algorithms
prange = PrangeISD(max_iterations=1000)
stern = SternISD(max_iterations=100, p=2, l=4)

# Create an experiment with the algorithms
experiment = Experiment([prange, stern])

# Run the experiment
n, k, t = 24, 12, 2
num_trials = 10
results = experiment.run_experiment(n, k, t, num_trials)

# Analyze and visualize the results
analysis = experiment.analyze_results()
fig, axes = experiment.plot_summary()
```

## Examples

See the `examples` directory for complete examples:

- `isd_comparison.py`: Compare Prange's and Stern's algorithms on random instances

## ISD Algorithms

### Prange's Algorithm

Prange's algorithm is one of the simplest ISD algorithms. It works by:
1. Randomly selecting k columns of H to form an information set
2. Attempting to put the submatrix of H into a form where it's invertible
3. If successful, computing e = (0, ..., 0, s'_1, ..., s'_{n-k})
4. Checking if the weight of e is <= t

### Stern's Algorithm

Stern's algorithm improves on Prange's by using a meet-in-the-middle approach:
1. Randomly selecting a set of k+l columns of H (where l is a parameter)
2. Transforming H into a form [I | M] using Gaussian elimination
3. Splitting the columns of M into two sets
4. Building sets of all linear combinations of p columns from each set
5. Looking for matches that result in a syndrome match
6. Constructing and checking candidate error vectors

## References

- Prange, E. (1962). The use of information sets in decoding cyclic codes. IRE Transactions on Information Theory, 8(5), 5-9.
- Stern, J. (1989). A method for finding codewords of small weight. In Coding Theory and Applications (pp. 106-113). Springer.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 