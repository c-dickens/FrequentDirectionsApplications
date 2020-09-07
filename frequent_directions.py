import numpy as np

class FrequentDirections:
    '''
    Implements the fast frequent directions algorithm using the doubled space technique.
    '''
    def __init__(self, d:int, sketch_size:int):
        self.d = d
        self.sketch_size = sketch_size
        self.m = 2*self.sketch_size
        self._sketch = np.zeros((self.m,self.d),dtype=float) # the sketch we populate
        self.zero_flag = True
        self.next_zero_id = 0

    def fit(self, X):
        '''
        Given dataset X, compute the frequent directions summary and return the sketch.
        '''
        n = X.shape[0]
        for i in range(n):
            Xi = X[i,:]

            # Insert row into the sketch
            if self.next_zero_id < self.m-1:
                self._sketch[self.next_zero_id,:] = Xi
                self.next_zero_id += 1
            # if the appended row is in final slot the rotate and reduce
            # elif self.next_zero_id >= self.m - 1:
            else: # self.next_zero_id should never exceed self.m-1
                self.__rotate_and_reduce__()
        self.sketch = self._sketch[:self.sketch_size,:]
                
            
    def __rotate_and_reduce__(self):
        '''
        Performs shrinkage via SVD and reduces the number of nonzero rows in the sketch
        '''
        _, S, Vt = np.linalg.svd(self._sketch,full_matrices=False)
        delta = S[self.sketch_size-1]**2 # 0-based indexing is used.
        S_reduce = np.sqrt(S[:self.sketch_size]**2 - delta)
        self._sketch[:self.sketch_size,:] =  S_reduce[:,None]*Vt[:self.sketch_size,:] # Use broadcasting for diag multiplication
        self._sketch[self.sketch_size:,:] = 0.0
        self.next_zero_id = self.sketch_size

    def get_covariance_error_bound(self,data,k):
        '''
        Evaluates the covariance error bound on input ``data'' at rank k.

        Note that this only applies when data == X as used in the fit function.
        '''
        U, S, Vt = np.linalg.svd(data, full_matrices=False)
        Xk = U[:,:k]@(np.diag(S[:k])@Vt[:k,:])
        delta_k = np.linalg.norm(data - Xk,ord='fro')**2
        return delta_k / (self.sketch_size - k )



    

