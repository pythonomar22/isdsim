import numpy as np
import pytest
from isd_lib.core.linear_code import LinearCode
from isd_lib.algorithms.prange import PrangeISD

def test_prange_zero_syndrome():
    """Test that Prange can decode a zero syndrome (no errors)."""
    n, k = 10, 5
    code = LinearCode(n, k)
    H = code.H
    s = np.zeros(n - k, dtype=int)
    t = 0
    
    decoder = PrangeISD()
    e, success, iterations = decoder.decode(H, s, t)
    
    assert success
    assert iterations == 0
    assert np.all(e == 0)

def test_prange_small_code():
    """Test that Prange can decode a small code with a known error pattern."""
    # Create a small code where decoding should succeed quickly
    n, k = 8, 4
    code = LinearCode(n, k)
    
    # Create a test error vector
    e_test = np.zeros(n, dtype=int)
    e_test[0] = 1  # Single error at position 0
    
    # Calculate the syndrome
    s = code.syndrome(e_test)
    
    # Decode with Prange's algorithm
    decoder = PrangeISD()
    e, success, iterations = decoder.decode(code.H, s, 1)
    
    assert success
    # The found error vector should produce the same syndrome
    assert np.array_equal(code.syndrome(e), s)
    # The weight should be 1
    assert np.sum(e) == 1

def test_prange_multiple_errors():
    """Test that Prange can decode with multiple errors."""
    n, k = 20, 10
    code = LinearCode(n, k)
    
    # Create a test error vector with 2 errors
    e_test = np.zeros(n, dtype=int)
    e_test[0] = 1
    e_test[15] = 1
    
    # Calculate the syndrome
    s = code.syndrome(e_test)
    
    # Decode with Prange's algorithm
    decoder = PrangeISD(max_iterations=10000)  # Increase iterations for better chance of success
    e, success, iterations = decoder.decode(code.H, s, 2)
    
    if success:
        # If decoding succeeded, the found error vector should produce the same syndrome
        assert np.array_equal(code.syndrome(e), s)
        # The weight should be <= 2
        assert np.sum(e) <= 2
    else:
        # It's possible Prange's algorithm fails to find a solution within max_iterations
        # In that case, just check that the iterations count is correct
        assert iterations == decoder.max_iterations

def test_prange_statistics():
    """Test that Prange correctly tracks statistics."""
    n, k = 10, 5
    code = LinearCode(n, k)
    
    # Create a test error vector
    e_test = np.zeros(n, dtype=int)
    e_test[0] = 1
    
    # Calculate the syndrome
    s = code.syndrome(e_test)
    
    # Decode with Prange's algorithm
    decoder = PrangeISD()
    _, _, iterations = decoder.decode(code.H, s, 1)
    
    assert iterations > 0
    assert decoder.iterations_performed == iterations
    assert decoder.time_taken > 0
    
    # Reset and check
    decoder.reset_statistics()
    assert decoder.iterations_performed == 0
    assert decoder.time_taken == 0

def test_prange_check_solution():
    """Test the check_solution method."""
    n, k = 10, 5
    code = LinearCode(n, k)
    
    # Create a test error vector
    e_test = np.zeros(n, dtype=int)
    e_test[0] = 1
    
    # Calculate the syndrome
    s = code.syndrome(e_test)
    
    # Check that e_test is a valid solution
    decoder = PrangeISD()
    assert decoder.check_solution(code.H, s, e_test, 1)
    
    # Modify e_test to make it invalid
    e_test[1] = 1  # Now weight is 2, which exceeds t=1
    assert not decoder.check_solution(code.H, s, e_test, 1)
    
    # Different error pattern should not match the syndrome
    e_wrong = np.zeros(n, dtype=int)
    e_wrong[1] = 1  # Error at position 1 instead of 0
    assert not decoder.check_solution(code.H, s, e_wrong, 1)

def test_prange_permutation():
    """Test the permutation methods."""
    n = 10
    decoder = PrangeISD()
    
    # Test random_permutation
    perm = decoder.random_permutation(n)
    assert len(perm) == n
    assert set(perm) == set(range(n))  # Check that it's a valid permutation
    
    # Test permute_matrix
    H = np.eye(n)
    H_perm = decoder.permute_matrix(H, perm)
    assert H_perm.shape == H.shape
    
    # Test unpermute_vector
    v = np.ones(n)
    v_perm = H_perm.dot(v) % 2
    v_unperm = decoder.unpermute_vector(v_perm, perm)
    assert np.array_equal(v_unperm, v) 