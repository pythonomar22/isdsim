import numpy as np
import time
from .base_isd import BaseISD

class PrangeISD(BaseISD):
    """
    Implementation of Prange's Information Set Decoding (ISD) algorithm.
    
    Prange's algorithm is one of the simplest ISD algorithms. It works by:
    1. Randomly selecting k columns of H to form an information set
    2. Attempting to put the submatrix of H corresponding to the information set into a form where
       it's invertible (using Gaussian elimination)
    3. If successful, computing e = (0, ..., 0, s'_1, ..., s'_{n-k})
    4. Checking if the weight of e is <= t
    
    If steps 2-4 fail, the algorithm restarts with a new random information set.
    """
    
    def decode(self, H, s, t):
        """
        Decode a syndrome to find an error vector using Prange's algorithm.
        
        Args:
            H (numpy.ndarray): Parity-check matrix
            s (numpy.ndarray): Syndrome
            t (int): Target error weight
            
        Returns:
            tuple: (error_vector, success_flag, iterations)
                - error_vector (numpy.ndarray): The found error vector, or None if failed
                - success_flag (bool): True if decoding was successful, False otherwise
                - iterations (int): Number of iterations performed
        """
        if s is None or np.all(s == 0):
            return np.zeros(H.shape[1], dtype=int), True, 0
        
        # Initialize
        n_rows, n_cols = H.shape
        k = n_cols - n_rows
        
        self.iterations_performed = 0
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            self.iterations_performed = iteration + 1
            
            # Step 1: Randomly select a permutation
            perm = self.random_permutation(n_cols)
            H_perm = self.permute_matrix(H, perm)
            
            # Split H_perm into two parts: H_I (first k columns) and H_P (remaining n-k columns)
            # We want to put H_P into systematic form [I_{n-k} | *]
            H_I = H_perm[:, :k]
            H_P = H_perm[:, k:]
            
            # Step 2: Perform Gaussian elimination on H_P
            # If H_P cannot be put into a form with an invertible (n-k)x(n-k) submatrix,
            # we'll get a smaller rank than n-k, and we need to try again
            _, s_prime, rank, _ = self.gaussian_elimination(H_P, s)
            
            if rank < n_rows:
                # H_P is not full rank, try again
                continue
                
            # Step 3: Construct the candidate error vector
            e_perm = np.zeros(n_cols, dtype=int)
            e_perm[k:] = s_prime  # Last n-k positions
            
            # Step 4: Check the weight
            weight = np.sum(e_perm)
            if weight <= t:
                # Convert back to the original ordering
                e = self.unpermute_vector(e_perm, perm)
                
                # Double-check the solution
                if self.check_solution(H, s, e, t):
                    self.time_taken = time.time() - start_time
                    return e, True, self.iterations_performed
        
        # If we reach here, we've failed to find a solution
        self.time_taken = time.time() - start_time
        return None, False, self.iterations_performed 