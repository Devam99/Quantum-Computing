import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

def validate_inputs(A, b):
    """Check if A is Hermitian, square, dimensions of power 2, and b has a matching dimension."""
    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian.")
    if np.log2(N) != int(np.log2(N)):
        raise ValueError(f"Dimension N={N} must be a power of 2.")
    if b.shape[0] != N:
        raise ValueError(f"b has dimension {b.shape[0]}, expected {N}.")
    if np.linalg.det(A) == 0:
        raise ValueError("A must be invertible.")

def normalise(v):
    """Return normalised vector v."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("v must be positive.")
    return v / norm

def classical_solve(A, b):
    """Solve Ax = b classically and return the normalised solution."""
    x = np.linalg.solve(A, b)
    return normalise(x)

def get_eigenvalues(A):
    """Return eigenvalues and eigenvectors of A."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return eigenvalues, eigenvectors

def choose_t0(eigenvalues, n_clock):
    """Choose the time parameter t0 so that eigenvalues map to."""