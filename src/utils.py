# Utils
import numpy as np

# Pauli operators
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z, 'II': np.kron(I,I),
              'IZ': np.kron(I,Z), 'ZI': np.kron(Z,I), 'ZZ': np.kron(Z,Z),
              'XX': np.kron(X,X), 'YY': np.kron(Y,Y)}

class Operators:

    '''Contains operators '''

    def __init__(self, n_qubit) -> None:
        self.n_qubit = n_qubit
        self.I = np.eye(2)
        self.H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.S = np.array([[1, 0], [0, 1j]]) # the phase gate
    
        if n_qubit == 2:
            self.cnot_01 = np.array([[1, 0, 0, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 0, 1], 
                                    [0, 0, 1, 0]])
            
            self.cnot_10 = np.array([[1, 0, 0, 0], 
                                    [0, 0, 0, 1], 
                                    [0, 0, 1, 0], 
                                    [0, 1, 0, 0]])
            
            self.swap = np.array([[1, 0, 0, 0], 
                                [0, 0, 1, 0], 
                                [0, 1, 0, 0], 
                                [0, 0, 0, 1]]) 
    
    def operate(self, operator, indx):
        ''' returns the operator (matrix) to operate on a single qubit on the given 
         index in the n-qubit system'''
        result = np.eye(2**indx)
        result = np.kron(result, operator)
        for _ in range(int(self.n_qubit-indx-len(operator)/2)):
            result = np.kron(result, np.eye(2))
        
        return result
    

def pauli_sum(lst):
    """
    Computes the Pauli sum of a list of Pauli operators.
    """
    result = 0
    for item in lst:
        if len(item)!= 2:
            raise ValueError(f'must be of length 2, not {len(item)}')
        if item[0] not in pauli_dict.keys():
            raise ValueError(f'Invalid Pauli operator: {item[0]}')
        
        result += item[1] * pauli_dict[item[0]] 

    return result

def entropy(H, level=0):
    '''computes the von Neumann entropies for subsystem A and subsystem B
    input: 
        H: the Hamiltonian matrix (with lmb already included)
    return:
        level: int, for which energy level to compute the entropy, default is 0
        S_A, S_B, the von Neumann entropies for subsystem A and subsystem B'''
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    def _compute_entropy(rho):
            eigenvalues, _ = np.linalg.eig(rho)
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
            return entropy
    # index of the lowest eigenvalue and eigenvector
    
    sorted_index = np.argsort(eigenvalues)
    if type(level) == int:
        level = [level]

    entropy_A = np.zeros(len(level))
    entropy_B = np.zeros(len(level))
    for i in level:
        alpha = eigenvectors[:, sorted_index[i]]

        # density matrix for the lowest energy state
        rho_0 = np.outer(alpha, np.conj(alpha))

        # partial trace of both of the subsystems
        rho_A = np.einsum("ijkl,jl->ik", rho_0.reshape(2, 2, 2, 2), np.eye(2))
        rho_B = np.einsum("ijkl,ik->jl", rho_0.reshape(2, 2, 2, 2), np.eye(2))
        
        entropy_A[i] = _compute_entropy(rho_A)
        entropy_B[i] = _compute_entropy(rho_B)

    # Print the von Neumann entropies for subsystem A and subsystem B
    return entropy_A, entropy_B

