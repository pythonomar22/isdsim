import numpy as np
import time
from .base_isd import BaseISD

class SternISD(BaseISD):
    """
    Implementation of Stern's Information Set Decoding (ISD) algorithm.
    
    Stern's algorithm improves on Prange's by using a meet-in-the-middle approach.
    It splits the error vector into two parts and looks for matches.
    
    The algorithm works by:
    1. Randomly selecting a set of k+l columns of H (where l is a parameter)
    2. Transforming H into a form [I | M] using Gaussian elimination
    3. Splitting the columns of M into two sets
    4. Building sets of all linear combinations of p columns from each set
    5. Looking for matches (collisions) that result in a syndrome match
    6. Constructing and checking candidate error vectors
    """
    
    def __init__(self, max_iterations=100, p=2, l=20):
        """
        Initialize the Stern ISD algorithm.
        
        Args:
            max_iterations (int): Maximum number of iterations before giving up
            p (int): Number of columns to use in combinations (both halves)
            l (int): Number of additional columns to include in the information set
        """
        super().__init__(max_iterations)
        self.p = p
        self.l = l
    
    def decode(self, H, s, t):
        """
        Decode a syndrome to find an error vector using Stern's algorithm.
        
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
            
            # Step 1: Randomly select k+l positions for the information set
            perm = self.random_permutation(n_cols)
            H_perm = self.permute_matrix(H, perm)
            
            # We want to transform H_perm into a form [I | M]
            # First k+l columns correspond to the information set
            # Remaining n-(k+l) columns correspond to the redundant part
            h = n_rows  # Shorthand for clarity
            
            # Split H_perm into H_I (first k+l columns) and H_P (remaining n-(k+l) columns)
            H_I = H_perm[:, :(k+self.l)]
            H_P = H_perm[:, (k+self.l):]
            
            # Perform Gaussian elimination to get H_P in row echelon form
            # and also transform s accordingly
            H_reduced, s_reduced, rank, pivots = self.gaussian_elimination(H_perm, s, full=True)
            
            if rank < h:
                # H_perm couldn't be transformed fully, try again
                continue
            
            # Extract the matrix A (shorthand for H_reduced)
            A = H_reduced
            
            # The matrix should now be in the form [I | M]
            # where I is an h x h identity matrix and M is an h x (n-h) matrix
            
            # Verify that the left part is indeed an identity matrix
            if not np.array_equal(A[:h, :h], np.eye(h)):
                continue
            
            # Extract M from A = [I | M]
            M = A[:h, h:]
            
            # Split M into two equal parts (approximately)
            mid = M.shape[1] // 2
            M1 = M[:, :mid]
            M2 = M[:, mid:]
            
            # Step 2: Build sets of all linear combinations of p columns from each half
            # For efficiency, we'll store (hash, indices) pairs
            set1 = {}
            
            # Generate all combinations of p columns from the first half
            for indices in self._combinations(mid, self.p):
                # Calculate the sum of the columns (in GF(2))
                column_sum = np.zeros(h, dtype=int)
                for idx in indices:
                    column_sum = (column_sum + M1[:, idx]) % 2
                
                # Convert to a hashable form (tuple) and store
                hash_value = tuple(column_sum)
                if hash_value in set1:
                    set1[hash_value].append(indices)
                else:
                    set1[hash_value] = [indices]
            
            # Step 3: Look for matches that result in syndrome s_reduced
            for indices in self._combinations(M2.shape[1], self.p):
                # Calculate the sum of the columns (in GF(2))
                column_sum = np.zeros(h, dtype=int)
                for idx in indices:
                    column_sum = (column_sum + M2[:, idx]) % 2
                
                # Calculate the target hash: s_reduced + column_sum (in GF(2))
                target = tuple((s_reduced[:h] + column_sum) % 2)
                
                # Check if we have a match in set1
                if target in set1:
                    # For each matching set of indices from set1
                    for indices1 in set1[target]:
                        # Construct the candidate error vector (in permuted form)
                        e_perm = np.zeros(n_cols, dtype=int)
                        
                        # Set the positions in the first half of M
                        for idx in indices1:
                            e_perm[h + idx] = 1
                        
                        # Set the positions in the second half of M
                        for idx in indices:
                            e_perm[h + mid + idx] = 1
                        
                        # Calculate the remaining part of the error vector
                        # e_I = H_I^-1 * (s - H_P * e_P)
                        e_P = e_perm[h:]
                        H_P_e_P = np.dot(H_perm[:, h:], e_P) % 2
                        remaining = (s - H_P_e_P) % 2
                        
                        # Use A (which is H in row echelon form) to solve for e_I
                        e_I = np.zeros(h, dtype=int)
                        for i in range(h-1, -1, -1):
                            # Compute the sum of known elements
                            sum_known = 0
                            for j in range(i+1, h):
                                sum_known = (sum_known + A[i, j] * e_I[j]) % 2
                            
                            # Compute the bit value for position i
                            e_I[i] = (remaining[i] - sum_known) % 2
                        
                        e_perm[:h] = e_I
                        
                        # Convert back to original ordering
                        e = self.unpermute_vector(e_perm, perm)
                        
                        # Check the weight and solution validity
                        weight = np.sum(e)
                        if weight <= t and self.check_solution(H, s, e, t):
                            self.time_taken = time.time() - start_time
                            return e, True, self.iterations_performed
        
        # If we reach here, we've failed to find a solution
        self.time_taken = time.time() - start_time
        return None, False, self.iterations_performed
    
    def _combinations(self, n, p):
        """
        Generate all combinations of p elements from a set of n elements.
        
        Args:
            n (int): Number of elements
            p (int): Number of elements to choose
            
        Yields:
            list: A combination of p indices
        """
        # Recursive helper function
        def _combinations_helper(start, remaining, current):
            if remaining == 0:
                yield current.copy()
                return
            
            for i in range(start, n - remaining + 1):
                current.append(i)
                yield from _combinations_helper(i + 1, remaining - 1, current)
                current.pop()
        
        yield from _combinations_helper(0, p, []) 