import numpy as np

Z = (np.array([[1, 0], 
               [0, -1j]])) * (np.array([[0, -1j], 
                                        [1j, 0]])) * (np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                                                                [1/np.sqrt(2), -1/np.sqrt(2)]])) * (np.array([[1, 0], 
                                                                                                              [0, 1j]]))
print(Z)