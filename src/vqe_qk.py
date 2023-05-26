import numpy as np
from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import VQE
from qiskit.opflow import MatrixOp
from qiskit.utils import QuantumInstance

# Define the Lipkin model parameters
epsilon = 4
Omega = 0
c = 0
omega_z = 3
omega_x = 0.2

# Construct the Hamiltonian matrix
H_0 = np.array([[epsilon, 0], [0, epsilon]])
H_1 = np.array([[c, omega_x], [omega_z, 0]])
Hamiltonian = H_0 + H_1

# Convert the Hamiltonian to a qubit operator
qubit_op = MatrixOp(Hamiltonian)

# Define the ansatz circuit
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')

# Set up the VQE solver
backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend)
vqe = VQE(ansatz, quantum_instance=quantum_instance)

# Solve the Lipkin model using VQE
result = vqe.compute_minimum_eigenvalue(qubit_op)

# Retrieve the ground state energy and corresponding state
ground_state_energy = result.eigenvalue.real
ground_state = result.eigenstate

print("Ground State Energy:", ground_state_energy)
print("Ground State:", ground_state)
