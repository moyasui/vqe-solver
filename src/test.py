from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import Operator
from qiskit.algorithms import VQE
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import I, Z
from qiskit.providers.aer import QasmSimulator

def lipkin_model_energy_eigenvalues(J, W):
    # Construct the Lipkin model Hamiltonian operator
    num_qubits = 2 * J

    pauli_x = Operator.from_label('X')
    pauli_z = Operator.from_label('Z')

    hamiltonian = 0.0
    for i in range(num_qubits - 1):
        hamiltonian += pauli_x ^ pauli_x + W * (pauli_z ^ pauli_z)

    # Define the VQE ansatz circuit
    ansatz = RealAmplitudes(num_qubits)

    # Define the backend and quantum instance
    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024)

    # Create the VQE instance
    vqe = VQE(ansatz, quantum_instance=quantum_instance)

    # Compute the energy eigenvalues
    eigenvalues = vqe.compute_minimum_eigenvalue(hamiltonian).eigenvalue.real

    return eigenvalues

# Compute energy eigenvalues for J=1 and W=0
J = 1
W = 0
eigenvalues_J1 = lipkin_model_energy_eigenvalues(J, W)
print("Energy Eigenvalues (J=1):", eigenvalues_J1)

# Compute energy eigenvalues for J=2 and W=0
J = 2
eigenvalues_J2 = lipkin_model_energy_eigenvalues(J, W)
print("Energy Eigenvalues (J=2):", eigenvalues_J2)
