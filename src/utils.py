# Utils
import numpy as np

# Pauli operators
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

def pauli_sum(lst):
    """
    Computes the Pauli sum of a list of Pauli operators.
    """
    result = 0
    for item in lst:
        if len(item)!= 2:
            raise ValueError(f'must be of length 2, not {len(item)}')
        if item[0] not in ['X', 'Y', 'Z', 'I']:
            raise ValueError(f'Invalid Pauli operator: {item[0]}')
        
        result += item[1] * pauli_dict[item[0]] 

    return result