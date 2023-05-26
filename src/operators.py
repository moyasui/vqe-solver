# basic operators

import numpy as np


def H(qubits, idx):
    1/np.sqrt(2) * np.array([[1, 1], [1, -1]])

# Define the Pauli X operator
def X(qubits,idx):
    return np.array([[0, 1], [1, 0]]) @ qubits[idx]

# Define the Pauli Y operator
def Y(qubits,idx):
    return np.array([[0, -1j], [1j, 0]]) @ q

# Define the Pauli Z operator
def Z(qubits,idx):
    return np.array([[1, 0], [0, -1]]) @ q



