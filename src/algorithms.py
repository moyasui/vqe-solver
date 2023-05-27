# Algorithms for using base

from scipy.optimize import minimize

class Vqe():
        def __init__(self, ansatz, init_points, optimiser=None):
            '''ansatz: a parametriced circuit that takes parmas: theta and phi'''
            self.ansatz = ansatz
            self.init_points = init_points # has to match the number of the parameters in the ansatz
            try:
                ansatz(init_points)
            except ValueError:
                raise ValueError(f'The initial points ({init_points}) do not match the dimension of the ansatz.')

            if optimiser is None:
                self.minimize = minimize
                
        def _objective(self,params):
            qc = self.ansatz(params)
            # maybe needs to be changed to measurement based can actually take the 
            # expectation value
            energy = qc.state.conj() @ (self.H @ qc.state)
            return energy

        def expectation(self,num_shots=1024):
            pass

        def minimise_eigenvalue(self, hamiltonian, num_shots=1024):
            '''
            Rotates the parametrised circuit to find the minimised energy using classical 
            minimisation algorithms.
            Inputs:
            hamiltonian: a parametrised circuit that takes theta and phi, do not depend on lambda,
            num_shots: (int) number of shots,
            return: (float) minimised energy eigenvalues.'''
            self.H = hamiltonian
            result = self.minimize(self._objective, self.init_points)
            min_params = result.x
            min_energy = result.fun

            

            return min_params, min_energy

        


if __name__ == "__main__":
     print("Testing VQE")