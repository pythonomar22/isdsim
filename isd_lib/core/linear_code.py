import numpy as np

class LinearCode:
    """
    A class representing a linear code over GF(2).
    
    Attributes:
        n (int): Code length (number of bits in a codeword)
        k (int): Code dimension (number of bits in the original message)
        G (numpy.ndarray): Generator matrix (k x n)
        H (numpy.ndarray): Parity-check matrix ((n-k) x n)
    """
    
    def __init__(self, n, k, G=None, H=None, systematic=True):
        """
        Initialize a linear code with given parameters.
        
        Args:
            n (int): Code length
            k (int): Code dimension
            G (numpy.ndarray, optional): Generator matrix. If None, a random matrix is generated.
            H (numpy.ndarray, optional): Parity-check matrix. If None, it's derived from G.
            systematic (bool, optional): Whether to convert G to systematic form. Defaults to True.
        """
        self.n = n
        self.k = k
        
        # Generate a random generator matrix if none provided
        if G is None and H is None:
            self.G = self._generate_random_generator_matrix(systematic)
            self.H = self._derive_parity_check_matrix()
        elif G is not None:
            self.G = G
            if H is None:
                self.H = self._derive_parity_check_matrix()
            else:
                self.H = H
        else:  # H is not None but G is None
            self.H = H
            self.G = self._derive_generator_matrix()
    
    def _generate_random_generator_matrix(self, systematic=True):
        """
        Generate a random generator matrix.
        
        Args:
            systematic (bool): Whether to convert to systematic form.
            
        Returns:
            numpy.ndarray: A random generator matrix
        """
        if systematic:
            # Directly create a matrix in systematic form [I_k | P]
            G = np.zeros((self.k, self.n), dtype=int)
            
            # Set the left part to identity
            G[:, :self.k] = np.eye(self.k, dtype=int)
            
            # Set the right part to random values
            G[:, self.k:] = np.random.randint(0, 2, size=(self.k, self.n - self.k))
            
            return G
        else:
            # Create a random binary matrix
            max_attempts = 10
            for _ in range(max_attempts):
                G = np.random.randint(0, 2, size=(self.k, self.n))
                
                # Check if the matrix has full rank
                if np.linalg.matrix_rank(G % 2) == self.k:
                    return G
            
            # If we couldn't generate a full-rank matrix after max_attempts,
            # fall back to systematic form
            return self._generate_random_generator_matrix(systematic=True)
    
    def _to_systematic_form(self, G):
        """
        Convert a generator matrix to systematic form [I_k | P].
        
        Args:
            G (numpy.ndarray): Generator matrix
            
        Returns:
            numpy.ndarray: Generator matrix in systematic form
        """
        # Perform Gaussian elimination (over GF(2))
        G_sys = G.copy()
        
        # For each row
        for i in range(self.k):
            # Find a pivot in column i
            pivot_found = False
            for j in range(i, self.k):
                if G_sys[j, i] == 1:
                    # Swap rows if necessary
                    if j != i:
                        G_sys[[i, j]] = G_sys[[j, i]]
                    pivot_found = True
                    break
            
            if not pivot_found:
                # If no pivot found, try a different column
                for col in range(self.n):
                    if col == i:
                        continue
                    pivot_found = False
                    for j in range(i, self.k):
                        if G_sys[j, col] == 1:
                            # Swap rows and columns
                            if j != i:
                                G_sys[[i, j]] = G_sys[[j, i]]
                            G_sys[:, [i, col]] = G_sys[:, [col, i]]
                            pivot_found = True
                            break
                    if pivot_found:
                        break
                if not pivot_found:
                    raise ValueError("Could not convert matrix to systematic form")
            
            # Eliminate other 1s in this column
            for j in range(self.k):
                if j != i and G_sys[j, i] == 1:
                    G_sys[j] = (G_sys[j] + G_sys[i]) % 2
        
        # Ensure the left part is identity
        for i in range(self.k):
            if G_sys[i, i] != 1:
                raise ValueError("Failed to convert to systematic form")
            for j in range(self.k):
                if i != j and G_sys[i, j] != 0:
                    raise ValueError("Failed to convert to systematic form")
        
        return G_sys
    
    def _derive_parity_check_matrix(self):
        """
        Derive the parity-check matrix from the generator matrix.
        Assumes G is in systematic form G = [I_k | P].
        
        Returns:
            numpy.ndarray: The parity-check matrix H = [-P^T | I_(n-k)]
        """
        if self.G.shape != (self.k, self.n):
            raise ValueError(f"Generator matrix has wrong shape: {self.G.shape}, expected {(self.k, self.n)}")
        
        # Extract P from G = [I_k | P]
        P = self.G[:, self.k:]
        
        # Create H = [-P^T | I_(n-k)]
        # In GF(2), -P == P
        H = np.zeros((self.n - self.k, self.n), dtype=int)
        H[:, :self.k] = P.T
        H[:, self.k:] = np.eye(self.n - self.k, dtype=int)
        
        return H
    
    def _derive_generator_matrix(self):
        """
        Derive the generator matrix from the parity-check matrix.
        Assumes H is in systematic form H = [-P^T | I_(n-k)].
        
        Returns:
            numpy.ndarray: The generator matrix G = [I_k | P]
        """
        if self.H.shape != (self.n - self.k, self.n):
            raise ValueError(f"Parity-check matrix has wrong shape: {self.H.shape}, expected {(self.n - self.k, self.n)}")
        
        # Extract P^T from H = [-P^T | I_(n-k)]
        P_T = self.H[:, :self.k]
        
        # Create G = [I_k | P]
        G = np.zeros((self.k, self.n), dtype=int)
        G[:, :self.k] = np.eye(self.k, dtype=int)
        G[:, self.k:] = P_T.T
        
        return G
    
    def encode(self, message):
        """
        Encode a message using the generator matrix.
        
        Args:
            message (numpy.ndarray): A binary vector of length k
            
        Returns:
            numpy.ndarray: The encoded codeword of length n
        """
        if len(message) != self.k:
            raise ValueError(f"Message length ({len(message)}) doesn't match code dimension ({self.k})")
        
        # Multiply message by generator matrix (over GF(2))
        codeword = np.dot(message, self.G) % 2
        
        return codeword
    
    def syndrome(self, vector):
        """
        Calculate the syndrome of a vector.
        
        Args:
            vector (numpy.ndarray): A binary vector of length n
            
        Returns:
            numpy.ndarray: The syndrome vector of length (n-k)
        """
        if len(vector) != self.n:
            raise ValueError(f"Vector length ({len(vector)}) doesn't match code length ({self.n})")
        
        # Multiply vector by parity-check matrix (over GF(2))
        return np.dot(self.H, vector) % 2
    
    def is_codeword(self, vector):
        """
        Check if a vector is a valid codeword.
        
        Args:
            vector (numpy.ndarray): A binary vector of length n
            
        Returns:
            bool: True if the vector is a codeword, False otherwise
        """
        return np.all(self.syndrome(vector) == 0)
    
    @staticmethod
    def hamming_weight(vector):
        """
        Calculate the Hamming weight (number of non-zero elements) of a vector.
        
        Args:
            vector (numpy.ndarray): A binary vector
            
        Returns:
            int: The Hamming weight
        """
        return np.sum(vector)
    
    @staticmethod
    def hamming_distance(vector1, vector2):
        """
        Calculate the Hamming distance between two vectors.
        
        Args:
            vector1 (numpy.ndarray): First binary vector
            vector2 (numpy.ndarray): Second binary vector
            
        Returns:
            int: The Hamming distance
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have the same length")
        
        return np.sum(vector1 != vector2)
    
    def __repr__(self):
        return f"LinearCode(n={self.n}, k={self.k})" 