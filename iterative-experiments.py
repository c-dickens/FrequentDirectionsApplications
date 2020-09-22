from data_factory import DataFactory
from frequent_directions_regression import FDRidge
from random_projection_regression import RPRidge
from sklearn.linear_model import Ridge,LinearRegression
from plot_config import fd_params, rfd_params, rp_params, hs_params
import numpy as np
from math import floor
import matplotlib.pyplot as plt

def synthetic_iterative_experiment():
    gamma = 100.0
    n = 2**10
    d = 2**9
    eff_rank = int(floor(0.05*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank)
    X,y,_ = ds.shi_phillips_synthetic()
    #X /= np.linalg.norm(X,ord=2)

    # Optimal 
    # clf = Ridge(alpha=gamma,fit_intercept=False)
    # clf.fit(X,y)
    # x_opt = clf.coef_

    # Optimal solution
    H = X.T@X + gamma*np.eye(d)
    x_opt = np.linalg.solve(H,X.T@y)

    # Iterate the FD regression
    iterations = 5
    m = int(2**5)
    fdr = FDRidge(fd_dim=m,gamma=gamma)
    fdr.fit(X,y)
    _, all_x = fdr.iterate(X,y,iterations)

    rfdr = FDRidge(fd_dim=m,fd_mode='RFD',gamma=gamma)
    rfdr.fit(X,y)
    _, rfd_all_x = rfdr.iterate(X,y,iterations)

    rp_ihs = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
    _, rp_all_x = rp_ihs.iterate_single(X,y)

    ihs = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
    _, ihs_all_x = ihs.iterate_multiple(X,y)

    fd_errors = np.zeros(iterations)
    rfd_errors = np.zeros_like(fd_errors,dtype=float)
    rp_errors = np.zeros_like(fd_errors)
    ihs_errors = np.zeros_like(fd_errors)
    for it in range(iterations):
        err = np.linalg.norm(all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        rfd_err = np.linalg.norm(rfd_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        rp_err = np.linalg.norm(rp_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        ihs_err = np.linalg.norm(ihs_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        print(f'Iteration {it}\tFD:{err:.5E}\tRFD:{rfd_err:.5E}\tRP:{rp_err:.5E}\tIHS:{ihs_err:.5E}')
        fd_errors[it] = err
        rfd_errors[it] = rfd_err
        rp_errors[it] = rp_err
        ihs_errors[it] = ihs_err


    fig, ax = plt.subplots(figsize=(5,2.5))
    ax.plot(1+np.arange(iterations), fd_errors,label='FD')
    ax.plot(1+np.arange(iterations), rfd_errors,label='RFD')
    ax.plot(1+np.arange(iterations), rp_errors,label='IHS-Single')
    ax.plot(1+np.arange(iterations), ihs_errors,label='IHS-Multi')
    ax.legend()
    ax.set_yscale('log')
    plt.show()
    fig.savefig('figures/iterative-synthetic.pdf',dpi=150,bbox_inches='tight',pad_inches=None)


def synthetic_real_experiment(data_name,gamma_reg):
    gamma = gamma_reg
    n = 10000
    ds = DataFactory(n=n)
    if data_name == 'CoverType':
        X,y = ds.fetch_forest_cover()
        rff_features = True
    elif data_name == 'w8a':
        X,y = ds.fetch_w8a()
        rff_features = True
    else:
        X,y = ds.fetch_year_predictions()
        rff_features = True
    X, y = X[:n], y[:n]

    # Whether to fit fourier features
    if rff_features:
        X = ds.feature_expansion(X,n_extra_features=1024)
    d = X.shape[1]


    # Optimal solution
    H = X.T@X + gamma*np.eye(d)
    x_opt = np.linalg.solve(H,X.T@y)

    # Iterate the FD regression
    iterations = 10
    m = int(2**8)
    fdr = FDRidge(fd_dim=m,gamma=gamma)
    fdr.fit(X,y)
    _, all_x = fdr.iterate(X,y,iterations)

    rfdr = FDRidge(fd_dim=m,fd_mode='RFD',gamma=gamma)
    rfdr.fit(X,y)
    _, rfd_all_x = rfdr.iterate(X,y,iterations)

    rp_ihs = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
    _, rp_all_x = rp_ihs.iterate_single(X,y)

    # ihs = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
    # _, ihs_all_x = ihs.iterate_single(X,y)

    # Measurement arrays
    fd_errors = np.zeros(iterations)
    rfd_errors = np.zeros_like(fd_errors,dtype=float)
    rp_errors = np.zeros_like(fd_errors)
    ihs_errors = np.zeros_like(fd_errors)

    for it in range(iterations):
        err = np.linalg.norm(all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        rfd_err = np.linalg.norm(rfd_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        rp_err = np.linalg.norm(rp_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        #ihs_err = np.linalg.norm(ihs_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        print(f'Iteration {it}\tFD:{err:.5E}\tRFD:{rfd_err:.5E}\tRP:{rp_err:.5E}')#\tIHS:{ihs_err:.5E}')
        fd_errors[it] = err
        rfd_errors[it] = rfd_err
        rp_errors[it] = rp_err
        #ihs_errors[it] = ihs_err

    fig, ax = plt.subplots(figsize=(5,2.5))
    ax.plot(1+np.arange(iterations), fd_errors,label='FD', **fd_params)
    ax.plot(1+np.arange(iterations), rfd_errors,label='RFD', **rfd_params)
    ax.plot(1+np.arange(iterations), rp_errors,label='IHS-Single', **rp_params)
    #ax.plot(1+np.arange(iterations), ihs_errors,label='IHS-Multi')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    #plt.show()
    fname = 'figures/iterative-'+data_name+str(gamma)+'.pdf'
    fig.savefig(fname,dpi=150,bbox_inches='tight',pad_inches=None)
def main():
    datasets = ['w8a',]#'CoverType', 'YearPredictions']
    gammas = [10., 100., 1000.]
    for d in datasets:
        for g in gammas:
            synthetic_real_experiment(d,g)

if __name__ == '__main__':
    main()

