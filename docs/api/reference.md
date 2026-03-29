# API Reference

Complete API documentation for Atom.

## Main Classes

### AtomicDFTSolver

The main solver class for atomic DFT calculations.

```python
from atom import AtomicDFTSolver

solver = AtomicDFTSolver(
    atomic_number: int,
    xc_functional: str,
    **kwargs
)
```

#### Parameters

- `atomic_number` (int): Atomic number of the element
- `xc_functional` (str): Exchange-correlation functional name
  - Valid options: `"LDA_PZ"`, `"LDA_PW"`, `"GGA_PBE"`, `"SCAN"`, `"RSCAN"`, `"R2SCAN"`, `"HF"`, `"PBE0"`, `"EXX"`, `"RPA"`
- `domain_size` (float, optional): Size of the computational domain in Bohr
- `finite_elements` (int, optional): Number of finite elements
- `polynomial_order` (int, optional): Polynomial order for basis functions
- `mesh_type` (str, optional): Type of mesh (`"exponential"`, `"polynomial"`, `"uniform"`)

#### Methods

- `solve()`: Execute the SCF calculation and return results

#### Returns

`SCFResult` object containing:
- `total_energy`: Total energy in Ha
- `converged`: Whether the calculation converged
- `n_iterations`: Number of SCF iterations
- `density`: Electron density array
- And more...

## Data Management

### AtomicDataManager

Class for managing atomic DFT datasets.

```python
from atom.data import AtomicDataManager

manager = AtomicDataManager(
    data_root: str,
    scf_xc_functional: str,
    forward_pass_xc_functionals: List[str]
)
```

See the [Data Management documentation](../atom/data/README.md) for more details.

## Module Structure

- `atom.solver`: Main solver class
- `atom.scf`: SCF iteration components
- `atom.xc`: Exchange-correlation functionals
- `atom.mesh`: Finite element mesh
- `atom.data`: Data management and loading
- `atom.pseudo`: Pseudopotential handling
- `atom.utils`: Utility functions

For more detailed documentation, see the source code or individual module docstrings.
