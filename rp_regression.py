import numpy as np
#from utils import sparse_projection

class RPRidge:

    def __init__(self, sk_dim:int,alpha=1.0):
        """
        Approximate ridge regression using a sparse RP sketch.

        sk_dim (int) - the number of rows retained in the sketch.
        alpha : float - the regularisation parameter for ridge regression.
        """
        self.sk_dim       = sk_dim
        self.alpha        = alpha
        self.is_fitted    = False

    def _sparse_rp(self,data,sparsity=5):
        """
        """
        [n,d] = data.shape
        sketch = np.zeros((self.sk_dim ,n),dtype=float)
        for i in range(n):
            nnz_loc = np.random.choice(self.sk_dim ,size=sparsity,replace=False)
            nnz_sign = np.random.choice([-1,1],size=sparsity,replace=True)
            sketch[nnz_loc,i] = nnz_sign
        return (1./np.sqrt(sparsity))*sketch@data

    def _sketch_data_targets(self,data,targets):
        """
        Calls the sketch on the [data, target] pair.
        """
        X = np.c_[data, targets]
        SX = self._sparse_rp(X)
        return SX[:,:-1], SX[:,-1]

    # def _get_inv(self,sketch):
    #     """
    #     Returns the inverse of Hessian approximation in a more scalable way
    #     using Woodbury
    #     """
    #     m,d = sketch.shape
    #     Im = np.eye(m)
    #     Id = np.eye(d)
    #     BBt = sketch@sketch.T
    #     I_BBt_inv = np.linalg.pinv(Im + BBt)
     
    #     return (1/self.alpha)*(Id - self.B.T@( I_BBt_inv@self.B/self.alpha))

    def fit_classic(self,X,y,random_state=10):
        """
        Fits the ridge regression model with data X and targets y using classical sketch

        (A^T S^T S A + alpha I) x_hs = A^T S^T S b
        """

        d = X.shape[1]
        # 1. sketch the data
        #SASb = self._sparse_rp(np.c_[X,y])
        SA, Sb = self._sketch_data_targets(X,y )#  SASb[:,:-1], SASb[:,-1]
        #H = B.T@B + (self.alpha+a)*np.eye(d)
        H = SA.T@SA + self.alpha*np.eye(d)
        self.H_inv = np.linalg.pinv(H) #self._get_inv() #
        self.cs_coef_ = self.H_inv@(SA.T@Sb) #np.linalg.solve(H, X.T@y)
        self.is_fitted = True

    def fit_hessian_sketch(self,X,y,random_state=10):
        """
        Fits the ridge regression model with data X and targets y using Hessian sketch:
        (A^T S^T S A + alpha I) x_hs = A^T b
        """

        d = X.shape[1]
        # 1. sketch the data
        #SASb = self._sparse_rp(np.c_[X,y])
        #SA, Sb = SASb[:,:-1], SASb[:,-1]
        SA, Sb = self._sketch_data_targets(X,y )
        #H = B.T@B + (self.alpha+a)*np.eye(d)
        H = SA.T@SA + self.alpha*np.eye(d)
        self.H_inv = np.linalg.pinv(H) #self._get_inv() #
        self.hs_coef_ = self.H_inv@(X.T@y) #np.linalg.solve(H, X.T@y)
        self.is_fitted = True

    def fit_ihs(self,X,y,iterations=10,random_state=10):
        """
        Fits the ridge regression with data X and y using the iterated hessian sketch
        with a single sketching matrix.
        """
        d = X.shape[1]
        if not self.is_fitted:
            self.B,_ = self._sketch(X,method=self.fd_mode)
            #self.H = B.T@B + self.alpha*np.eye(d) # rough approx to X.T@X + self.alpha*np.eye(d)
            #self.H_inv = np.linalg.pinv(H) # || I -  \hat{H_inv} H_true ||_2
            self.H_inv = self._get_inv()
        SA = self._sparse_rp(X)
        H = SA.T@SA + self.alpha*np.eye(d)
        H_inv = np.linalg.pinv(H)
        w = np.zeros( (d,1))
        all_w = np.zeros((d,iterations))
        XTy = (X.T@y).reshape(-1,1)
        for it in range(iterations):
            grad = X.T@(X@w) + self.alpha*w - XTy
            w += - H_inv@grad
            all_w[:,it] = np.squeeze(w)
        return w, all_w

