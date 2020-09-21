import numpy as np
import datetime
from scipy import linalg

class FrequentDirections:
    def __init__(self,d,m=8):
        '''
        Class wrapper for all FD-type methods
        '''
        self.d = d
        self.reset(sketch_dim=m)
        self.delta = 0. # For RFD
        
    def reset(self,sketch_dim=None):
        '''
        Initialises or resets the sketch parameters
        '''
        if sketch_dim is not None:
            self.sketch_dim = sketch_dim
        self.sketch = np.zeros((self.sketch_dim,self.d),dtype=float)
        self.Vt = np.zeros((self.sketch_dim,self.d),dtype=float)
        self.S2 = np.zeros(self.sketch_dim,dtype=float) # singular values squared
        
    def fit(self,X,batch_size=1):
        '''
        Fits the FD transform to dataset X
        '''
        #self.sketch = np.concatenate()
        n = X.shape[0]
        count = 0
        extra_space = np.zeros_like(self.sketch)
        head = 0
        tail = batch_size
        for i in range(0,n,batch_size):
            batch = X[i:i+batch_size,:]
            aux = np.concatenate((self.sketch,batch),axis=0)
            #print(f'Start:{i}\tStop{i+batch_size}')
            #print(batch.shape)
            try:
                _, s, self.Vt = np.linalg.svd(aux, full_matrices=False)
            except np.linalg.LinAlgError:
                np.save(f'./np_svd_fail{datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}.npy', aux)
                _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver='gesvd')
            self.S2 = s**2 
            self._shrink()  # shrink self.s2 and self.Vt
            self.sketch = self.Vt * np.sqrt(self.S2).reshape(-1, 1)
        
    def get(self):
        return self.sketch, self.S2, self.Vt
        
            
class FastFrequentDirections(FrequentDirections):
    '''
    Implements the fast version of FD by doubling space
    '''
    
    def _shrink(self):
        self.S2 = self.S2[:self.sketch_dim] - self.S2[self.sketch_dim]
        self.Vt = self.Vt[:self.sketch_dim]
        
class iSVD(FrequentDirections):
    '''
    Implements the ***heuristic*** incremental SVD
    '''
    
    def _shrink(self):
        self.S2 = self.S2[:self.sketch_dim]
        self.Vt = self.Vt[:self.sketch_dim]
        
class RobustFrequentDirections(FrequentDirections):
    '''
    Implements the RFD version of FD 
    '''
    
    def _shrink(self):
        self.delta += self.S2[self.sketch_dim]/2.
        self.S2 = self.S2[:self.sketch_dim] - self.S2[self.sketch_dim]
        self.Vt = self.Vt[:self.sketch_dim]





# class FrequentDirections:
#     '''
#     Implements the fast frequent directions algorithm using the doubled space technique.
#     '''
#     def __init__(self, d:int, sketch_size:int):
#         self.d = d
#         self.sketch_size = sketch_size
#         self.m = 2*self.sketch_size
#         self._sketch = np.zeros((self.m,self.d),dtype=float) # the sketch we populate
#         self.zero_flag = True
#         self.next_zero_id = 0

#     def fit(self, X):
#         '''
#         Given dataset X, compute the frequent directions summary and return the sketch.
#         '''
#         n = X.shape[0]
#         for i in range(n):
#             Xi = X[i,:]

#             # Insert row into the sketch
#             if self.next_zero_id < self.m-1:
#                 self._sketch[self.next_zero_id,:] = Xi
#                 self.next_zero_id += 1
#             # if the appended row is in final slot the rotate and reduce
#             # elif self.next_zero_id >= self.m - 1:
#             else: # self.next_zero_id should never exceed self.m-1
#                 self.__rotate_and_reduce__()
#         self.sketch = self._sketch[:self.sketch_size,:]
                
            
#     def __rotate_and_reduce__(self):
#         '''
#         Performs shrinkage via SVD and reduces the number of nonzero rows in the sketch
#         '''
#         # if np.isnan(self._sketch).any():
#         #     print('NaN detected!')
#         _, S, Vt = np.linalg.svd(self._sketch,full_matrices=False)
#         delta = S[self.sketch_size-1]**2 # 0-based indexing is used.
#         S_reduce = S[:self.sketch_size]**2 - delta
#         S_reduce[np.abs(S_reduce < 1E-10)] = 0.0 # precision can go negative
#         # Check all shrunk params are large enough 
#         # print((S[:self.sketch_size]**2 - delta)[:,None])
        
#         assert (S_reduce >= 0.0).all() 
#         # S_reduce = np.sqrt(S[:self.sketch_size]**2 - delta)
#         S_reduce = np.sqrt(S_reduce)
#         self._sketch[:self.sketch_size,:] =  S_reduce[:,None]*Vt[:self.sketch_size,:] # Use broadcasting for diag multiplication
#         self._sketch[self.sketch_size:,:] = 0.0
        
#         self.next_zero_id = self.sketch_size

#     def get_covariance_error_bound(self,data,k):
#         '''
#         Evaluates the covariance error bound on input ``data'' at rank k.

#         Note that this only applies when data == X as used in the fit function.
#         '''
#         U, S, Vt = np.linalg.svd(data, full_matrices=False)
#         Xk = U[:,:k]@(np.diag(S[:k])@Vt[:k,:])
#         delta_k = np.linalg.norm(data - Xk,ord='fro')**2
#         return delta_k / (self.sketch_size - k )



    

