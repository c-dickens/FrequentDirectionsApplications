import numpy as np
from data_factory import DataFactory
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from fd_regression import FrequentDirectionRidge
from rp_regression import RPRidge
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
np.set_printoptions(precision=4)

def main():

    alpha = 1.
    #X,y = make_regression(n_samples=1000,n_features=20,noise=1.0,n_informative=5,tail_strength=0.1,random_state=10)
    X,y = DataFactory(n=1000,d=250,random_seed=100).fetch_low_rank_ridge(effective_rank=25,tail_strength=0.1)
    X = StandardScaler().fit_transform(X)
    X /= np.linalg.norm(X,ord='fro')
    _,d = X.shape
    

    _,S,_ = np.linalg.svd(X,full_matrices=False)
    fig,ax = plt.subplots(dpi=125)
    ax.plot(S)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    fig.savefig('figures/ridge_sing_val_profile.pdf')
    # plt.show()

    clf = Ridge(alpha=alpha)
    clf.fit(X,y)
    

    x_rr = clf.coef_

    # Test the sketching method.
    iterations = 10
    sk_dim = 50
    fd_clf = FrequentDirectionRidge(fd_dim=sk_dim,alpha=alpha)
    fd_clf.fit(X,y)
    w_ihs, w_all_ihs = fd_clf.iterate(X,y,iterations)
    fd_error = fd_clf.sketch_error_bound(X,25)
    print(f'FD Error bound: {fd_error:.4f}')
    x_fd_rr = fd_clf.coef_
  
    print(f'Error:{(np.linalg.norm(x_fd_rr - x_rr)/np.linalg.norm(x_rr)):.4f}')


    fd_iteration_errors = []
    for i in range(iterations):
        ww = w_all_ihs[:,i]
        error = np.linalg.norm(x_rr - ww)
        if error < 1E-15:
            w_ihs = ww
            print(f'Converged after {i} iterations.')
            break
        fd_iteration_errors.append(error)
        print(f'Iteration: {i} \t Error: {error:.2E}')

    print(f'||x_rr||={np.linalg.norm(x_rr):.4f}')
    print(f'||x_fd||={np.linalg.norm(x_fd_rr):.4f}')
    print(f'||x_fd_ihs||={np.linalg.norm(w_ihs):.4f}')
    # print(np.c_[w_ihs, x_rr, x_fd_rr])

    rp_ridge = RPRidge(sk_dim,alpha=alpha)
    rp_ridge.fit_classic(X,y)
    rp_ridge.fit_hessian_sketch(X,y)
    rp_ridge.fit_ihs(X,y)
    x_cs = rp_ridge.cs_coef_
    x_hs = rp_ridge.hs_coef_
    x_rp_ihs, x_ihs_all = rp_ridge.fit_ihs(X,y)
    print(f'||x_cs||={np.linalg.norm(x_cs):.4f}')
    print(f'||x_hs||={np.linalg.norm(x_hs):.4f}')
    print(f'Classic error: {np.linalg.norm(x_cs - x_rr):.4f}')
    print(f'Hessian Sketch error: {np.linalg.norm(x_hs - x_rr):.4f}')

    rp_iteration_errors = []
    for i in range(iterations):
        xx = x_ihs_all[:,i]
        error = np.linalg.norm(x_rr - xx)
        if error < 1E-15:
            w_ihs = xx
            print(f'Converged after {i} iterations.')
            break
        rp_iteration_errors.append(error)
        print(f'Iteration: {i} \t Error: {error:.2E}')

    fig, ax = plt.subplots(dpi=125)
    ax.plot(1+np.arange(len(fd_iteration_errors)), fd_iteration_errors,label='FD')
    ax.plot(1+np.arange(len(rp_iteration_errors)), rp_iteration_errors,label='RP')
    #ax.set_xlim(0)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$ \log ||x^t - x^*||$')
    plt.show()
    fig.savefig('figures/iterative-sketching.pdf',bbox_inches='tight')#,pad_inches=0)

    
     

if __name__ == '__main__':
    main()
