import numpy as np



# Parameters
Hx = 2.0
Hz = 3.0
lambda_value = 0

# Hamiltonian matrices
H0_coeff = np.array([0.0, 2.5, 6.5, 7.0])
H0 = np.diag(H0_coeff)
H1 = np.array([[Hz, 0, 0, 0],
               [0, -Hz, Hx, 0],
               [0, Hx, -Hz, 0],
               [Hx, 0, 0, Hz]])

# Construct full Hamiltonian H = H0 + lambda_value * H1
H = H0 + lambda_value * H1

# entropy(lambda_value)