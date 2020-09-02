import numpy as np
from frequentDirections import FrequentDirections

class FrequentDirectionRidge:

    def __init__(self, fd_dim:int,fd_mode='FD',alpha=1.0):
        """
        Approximate ridge regression using the FD sketch.

        fd_dim (int) - the number of rows retained in the FD sketch.
        fd_mode (str) : mode for frequent directions FD or RFD.
        alpha : float - the regularisation parameter for ridge regression.
        """
        self.fd_dim       = fd_dim
        self.fd_mode      = fd_mode
        self.alpha        = alpha
        self.is_fitted    = False


    def fit(self,X,y):
        """
        Fits the ridge regression model with data X and targets y
        """

        d = X.shape[1]
        # 1. sketch the data
        self.B,a = self._sketch(X,method=self.fd_mode)
        #H = B.T@B + (self.alpha+a)*np.eye(d)
        #self.H = H
        self.H_inv = self._get_inv() #np.linalg.pinv(H)
        self.coef_ = self.H_inv@(X.T@y) #np.linalg.solve(H, X.T@y)
        self.is_fitted = True

    def iterate(self,X,y,iterations=10):
        """
        Perform `iterations` number of Newton steps for the regression problem.

        H_true = A.T@A + alphe*np.eye(d)

        Think I can show 
        || I -  \hat{H_inv} H_true ||_2 <= eps ||A - A_k||_F^2 / alpha (can be made < 1)
        """
        d = X.shape[1]
        if not self.is_fitted:
            self.B,_ = self._sketch(X,method=self.fd_mode)
            #self.H = B.T@B + self.alpha*np.eye(d) # rough approx to X.T@X + self.alpha*np.eye(d)
            #self.H_inv = np.linalg.pinv(H) # || I -  \hat{H_inv} H_true ||_2
            self.H_inv = self._get_inv()
        w = np.zeros( (d,1))
        all_w = np.zeros((d,iterations))
        XTy = (X.T@y).reshape(-1,1)
        for it in range(iterations):
            grad = X.T@(X@w) + self.alpha*w - XTy
            w += - self.H_inv@grad
            all_w[:,it] = np.squeeze(w)
        return w, all_w
        

    def _sketch(self, A, method='FD'):
        """
        Performs the FD sketch on array A
        """
        n,d = A.shape
        # This is where the sketching actually happens
        self.sketcher = FrequentDirections(d,self.fd_dim)
        for i in range(n):
            row = A[i,:]
            self.sketcher.append(row)
        sketch = self.sketcher.get()
        a = self.sketcher.alpha
        if method == 'RFD':
            return sketch,a
        return sketch, 0.

    def _get_inv(self):
        """
        Returns the inverse of Hessian approximation in a more scalable way
        using Woodbury
        """
        m,d = self.B.shape
        Im = np.eye(m)
        Id = np.eye(d)
        BBt = self.B@self.B.T
        I_BBt_inv = np.linalg.pinv(Im + BBt)
     
        return (1/self.alpha)*(Id - self.B.T@( I_BBt_inv@self.B/self.alpha))


    def sketch_error_bound(self,A, k):
        """
        Returns the error bound from the FD sketch for a rank-k approximation to A
        """
        return self.sketcher.error_bound(A,k)







