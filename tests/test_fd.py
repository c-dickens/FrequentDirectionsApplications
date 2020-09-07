import pytest
import numpy as np
from frequent_directions import FrequentDirections
from syntheticDataMaker import SyntheticDataMaker

def get_data(n,d,k,snr=10.0):
    '''
    Defines the dataset for the test setup
    '''
    dataMaker = SyntheticDataMaker()
    dataMaker.initBeforeMake(d, k, signal_to_noise_ratio=snr)    
    X = dataMaker.makeMatrix(n) 
    return X

def get_covariance_bound(X,k,sketch_dimension):
    '''
    Given input data X and rank k, evaluate tthe covariance error bound:
    ||X - Xk||_F^2 / (sketch_dimension - k)
    '''
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Xk = U[:,:k]@(np.diag(S[:k])@Vt[:k,:])
    delta_k = np.linalg.norm(X- Xk,ord='fro')**2
    return delta_k / (sketch_dimension - k )
    

def test_fd_shape():
    '''Ensures that the FD sketch returns the correct size sketch.
    '''
    n = 500
    d = 100
    k = 0
    ell = 50
    X = get_data(n,d,k)
    fd = FrequentDirections(d,ell)
    fd.fit(X)
    B = fd.sketch
    assert B.shape == (ell,d)
    
def test_covariance_error():
    '''
    Ensures that the covariance error is within the bound for a fixed
    index of the the rank to test.
    '''
    n = 500
    d = 100
    ell = 50
    k = 0
    X = get_data(n,d,k)
    fd = FrequentDirections(d,ell)
    fd.fit(X)
    B = fd.sketch
    cov_error = np.linalg.norm(B.T@B - X.T@X, ord=2)
    assert cov_error < np.linalg.norm(X)**2 / ell


def test_covariance_error_random_instance():
    '''
    Ensures that the covariance error is within the bound for a fixed
    index of the the rank to test.
    '''
    seed = 100
    np.random.seed(seed)
    n = 1000
    d = 200
    ell = 50
    k = np.random.randint(0,d//4,dtype=int)
    snr = 5.0

    X = get_data(n,d,k,snr)
    fd = FrequentDirections(d,ell)
    fd.fit(X)
    B = fd.sketch
    assert B.shape == (ell,d)
    cov_error = np.linalg.norm(B.T@B - X.T@X, ord=2)
    bound = fd.get_covariance_error_bound(X,k)#get_covariance_bound(X,k,ell)
    assert cov_error < bound