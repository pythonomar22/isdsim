import numpy as np
import pytest
from isd_lib.core.linear_code import LinearCode

def test_linear_code_init():
    # Test initialization with random generator matrix
    n, k = 10, 5
    code = LinearCode(n, k)
    assert code.n == n
    assert code.k == k
    assert code.G.shape == (k, n)
    assert code.H.shape == (n - k, n)
    
    # Check if G is in systematic form
    assert np.array_equal(code.G[:, :k], np.eye(k))
    
    # Check if H is in systematic form
    assert np.array_equal(code.H[:, k:], np.eye(n - k))
    
    # Check if G and H are orthogonal
    assert np.all((np.dot(code.H, code.G.T) % 2) == 0)

def test_encode_decode():
    n, k = 10, 5
    code = LinearCode(n, k)
    
    # Test encoding
    message = np.array([1, 0, 1, 0, 1])
    codeword = code.encode(message)
    assert len(codeword) == n
    
    # Check if encoded message is a codeword
    assert code.is_codeword(codeword)
    
    # Test syndrome
    syndrome = code.syndrome(codeword)
    assert np.all(syndrome == 0)
    
    # Add an error and check syndrome
    error_pos = 0
    codeword_with_error = codeword.copy()
    codeword_with_error[error_pos] = (codeword_with_error[error_pos] + 1) % 2
    
    assert not code.is_codeword(codeword_with_error)
    assert np.any(code.syndrome(codeword_with_error) != 0)

def test_hamming_weight_distance():
    # Test Hamming weight
    vector = np.array([1, 0, 1, 1, 0, 1])
    assert LinearCode.hamming_weight(vector) == 4
    
    # Test Hamming distance
    vector1 = np.array([1, 0, 1, 1, 0])
    vector2 = np.array([1, 1, 0, 1, 0])
    assert LinearCode.hamming_distance(vector1, vector2) == 2

def test_init_with_matrices():
    # Initialize with a specific generator matrix
    n, k = 6, 3
    G = np.array([
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1]
    ])
    code = LinearCode(n, k, G=G)
    
    # Check that H is derived correctly
    H_expected = np.array([
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1]
    ])
    assert np.array_equal(code.H, H_expected)
    
    # Initialize with a specific parity-check matrix
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1]
    ])
    code2 = LinearCode(n, k, H=H)
    
    # Check that G is derived correctly
    assert np.array_equal(code2.G, G)

def test_error_cases():
    n, k = 10, 5
    code = LinearCode(n, k)
    
    # Test encoding with wrong message length
    with pytest.raises(ValueError):
        code.encode(np.array([1, 0, 1]))
    
    # Test syndrome with wrong vector length
    with pytest.raises(ValueError):
        code.syndrome(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    
    # Test Hamming distance with vectors of different lengths
    with pytest.raises(ValueError):
        LinearCode.hamming_distance(np.array([1, 0, 1]), np.array([1, 0])) 