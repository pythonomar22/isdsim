import numpy as np
from abc import ABC, abstractmethod

class BaseISD(ABC):
    """
    Base class for Information Set Decoding (ISD) algorithms.
    
    This abstract class defines the interface for all ISD algorithm implementations.
    """
    
    def __init__(self, max_iterations=1000):
        """
        Initialize the ISD algorithm.
        
        Args:
            max_iterations (int): Maximum number of iterations before giving up.
        """
        self.max_iterations = max_iterations
        self.iterations_performed = 0
        self.time_taken = 0
    
    @abstractmethod
    def decode(self, H, s, t):
        """
        Decode a syndrome to find an error vector.
        
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
        pass
    
    @staticmethod
    def check_solution(H, s, e, t):
        """
        Check if an error vector is a valid solution to the syndrome decoding problem.
        
        Args:
            H (numpy.ndarray): Parity-check matrix
            s (numpy.ndarray): Syndrome
            e (numpy.ndarray): Error vector
            t (int): Target error weight
            
        Returns:
            bool: True if e is a valid solution, False otherwise
        """
        # Check if He^T = s
        computed_syndrome = np.dot(H, e) % 2
        syndrome_match = np.array_equal(computed_syndrome, s)
        
        # Check if weight(e) <= t
        weight_ok = np.sum(e) <= t
        
        return syndrome_match and weight_ok
    
    @staticmethod
    def random_permutation(n):
        """
        Generate a random permutation of n elements.
        
        Args:
            n (int): Number of elements
            
        Returns:
            numpy.ndarray: A random permutation of n elements
        """
        return np.random.permutation(n)
    
    @staticmethod
    def permute_matrix(H, permutation):
        """
        Permute the columns of a matrix according to a permutation.
        
        Args:
            H (numpy.ndarray): Matrix
            permutation (numpy.ndarray): Permutation
            
        Returns:
            numpy.ndarray: The permuted matrix
        """
        return H[:, permutation]
    
    @staticmethod
    def unpermute_vector(v, permutation):
        """
        Revert a permutation on a vector.
        
        Args:
            v (numpy.ndarray): Vector
            permutation (numpy.ndarray): Permutation
            
        Returns:
            numpy.ndarray: The unpermuted vector
        """
        result = np.zeros_like(v)
        for i, p in enumerate(permutation):
            result[p] = v[i]
        return result
    
    @staticmethod
    def gaussian_elimination(H, s=None, full=False):
        """
        Perform Gaussian elimination on a matrix over GF(2).
        
        Args:
            H (numpy.ndarray): Matrix
            s (numpy.ndarray, optional): Syndrome vector
            full (bool, optional): Whether to perform full Gaussian elimination or stop after finding a basis
            
        Returns:
            tuple: (H', s', rank, pivots)
                - H' (numpy.ndarray): Transformed matrix
                - s' (numpy.ndarray or None): Transformed syndrome (if s is provided)
                - rank (int): Rank of the matrix
                - pivots (list): List of pivot columns
        """
        H_copy = H.copy()
        s_copy = s.copy() if s is not None else None
        n_rows, n_cols = H_copy.shape
        
        rank = 0
        pivots = []
        
        for j in range(n_cols):
            # Find pivot in column j
            pivot_row = None
            for i in range(rank, n_rows):
                if H_copy[i, j] == 1:
                    pivot_row = i
                    break
            
            if pivot_row is not None:
                # Swap rows
                if pivot_row != rank:
                    H_copy[[rank, pivot_row]] = H_copy[[pivot_row, rank]]
                    if s_copy is not None:
                        s_copy[[rank, pivot_row]] = s_copy[[pivot_row, rank]]
                
                # Eliminate other 1s in this column
                for i in range(n_rows):
                    if i != rank and H_copy[i, j] == 1:
                        H_copy[i] = (H_copy[i] + H_copy[rank]) % 2
                        if s_copy is not None:
                            s_copy[i] = (s_copy[i] + s_copy[rank]) % 2
                
                pivots.append(j)
                rank += 1
                
                if rank == n_rows and not full:
                    break
        
        return H_copy, s_copy, rank, pivots
    
    def reset_statistics(self):
        """Reset the algorithm's statistics."""
        self.iterations_performed = 0
        self.time_taken = 0 