import sys
sys.path.append('..')
from data_factory import DataFactory
from frequent_directions import FrequentDirections, FastFrequentDirections,iSVD,RobustFrequentDirections
from frequent_directions_regression import FDRidge
from random_projection_regression import RPRidge
import numpy as np
from math import floor
import matplotlib.pyplot as plt

def test_one_shot():
    '''
    Tests the FD sketch for one-shot regression
    '''
    gamma = 1024
    n = 2**10
    d=2**8
    eff_rank = int(floor(0.25*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank)
    X,y,_ = ds.shi_phillips_synthetic()
    X /= np.linalg.norm(X,ord='fro')
    sketch_dimensions = np.array([16,32,64,128,200])
    fd_errors = np.zeros_like(sketch_dimensions,dtype=float)
    rfd_errors = np.zeros_like(sketch_dimensions,dtype=float)

    # Optimal solution
    H = X.T@X + gamma*np.eye(d)
    x_opt = np.linalg.solve(H,X.T@y)


    for i, si in enumerate(sketch_dimensions):
        # FD
        fdr = FDRidge(fd_dim=si,gamma=gamma)
        fdr.fit(X,y)
        x_fd = fdr.coef_
        fd_errors[i] = np.linalg.norm(x_fd - x_opt)/np.linalg.norm(x_opt)

        # RFD 
        rfdr = FDRidge(fd_dim=si,fd_mode='RFD',gamma=gamma)
        rfdr.fit(X,y)
        x_rfd = rfdr.coef_
        rfd_errors[i] = np.linalg.norm(x_rfd - x_opt)/np.linalg.norm(x_opt)
        print(f'FD Error\t{(fd_errors[i]):.4E}\t RFD Error\t{(rfd_errors[i]):.4E}')

def test_iterates():
    '''
    Tests the iterative procedure
    '''
    gamma = 1024
    n = 2**10
    d=2**8
    eff_rank = int(floor(0.5*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank)
    X,y,_ = ds.shi_phillips_synthetic()
    X /= np.linalg.norm(X,ord='fro')

    # Optimal solution
    H = X.T@X + gamma*np.eye(d)
    x_opt = np.linalg.solve(H,X.T@y)

    # Iterate the FD regression
    iterations = 10
    fdr = FDRidge(fd_dim=16,gamma=gamma)
    fdr.fit(X,y)
    _, all_x = fdr.iterate(X,y,iterations)

    rfdr = FDRidge(fd_dim=16,fd_mode='RFD',gamma=gamma)
    rfdr.fit(X,y)
    _, rfd_all_x = rfdr.iterate(X,y,iterations)
    for it in range(iterations):
        err = np.linalg.norm(all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        rfd_err = np.linalg.norm(rfd_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        print(f'Iteration {it}\tFD:{err:.3E}\tRFD:{rfd_err:.3E}')

def test_bias():
    n = 10**3
    d= 125
    eff_rank = int(floor(0.1*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank)
    # X,y,w0 = ds.shi_phillips_synthetic()
    # X /= np.linalg.norm(X,ord='fro')
    X,y,w0 = ds.mahoney_synthetic(noise_std=0.1)
    X /= np.linalg.norm(X,ord='fro')

    # Optimal bias
    mm = 64
    all_gammas = np.array([2**_ for _ in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]])
    H = X.T@X
    opt_bias = np.zeros_like(all_gammas,dtype=float)
    lower_bound = np.zeros_like(opt_bias)
    upper_bound = np.zeros_like(opt_bias)
    fd_bias_vals = np.zeros_like(opt_bias)
    rfd_bias_vals = np.zeros_like(opt_bias)
    rp_bias_vals = np.zeros_like(opt_bias)
    hs_bias_vals = np.zeros_like(opt_bias)
    for i,g in enumerate(all_gammas):
        Hg = H + g*np.eye(d)
        bias_g = (g**2)*np.linalg.norm(np.linalg.pinv(Hg)@w0)**2 #np.linalg.norm(x_opt - w0)**2 #
        #print(bias_g)
        
        # FD Sketch
        fdr = FDRidge(fd_dim=mm,gamma=g)
        fdr.fit(X,y)
        fd_bias = fdr.get_bias(X,w0) 
        fd_bias_sq = np.linalg.norm(fd_bias)**2 #np.linalg.norm(fdr.coef_ - w0)**2 #

        # RFD Sketch
        rfdr = FDRidge(fd_dim=mm,fd_mode='RFD',gamma=g)
        rfdr.fit(X,y)
        rfd_bias = rfdr.get_bias(X,w0) 
        rfd_bias_sq = np.linalg.norm(rfd_bias)**2 #np.linalg.norm(fdr.coef_ - w0)**2 #

        # RP sketch
        rp = RPRidge(rp_dim=mm,gamma=g)
        rp.fit_classical(X,y)
        rp_bias_sq = np.linalg.norm(rp.get_classical_bias(X,w0))**2

        # Hessian Sketch
        rp.fit_hessian_sketch(X,y)
        rp_hes_bias_sq = np.linalg.norm(rp.get_hessian_sketch_bias(X,w0))**2

        
        # Bounds 
        c = 1. - 1./(g*mm)
        if c < 0:
            continue
        upper = (1./c**2)*bias_g
        lower = (c**2)*bias_g
        opt_bias[i] = bias_g
        lower_bound[i] = lower 
        upper_bound[i] = upper
        fd_bias_vals[i] = fd_bias_sq
        rfd_bias_vals[i] = rfd_bias_sq
        rp_bias_vals[i] = rp_bias_sq
        hs_bias_vals[i] = rp_hes_bias_sq
        print(f'γ:{g:.6f} c:{c:.5f} OPT:{bias_g:.6f} Lower:{lower:.6f} Upper:{upper:.6f} FD:{fd_bias_sq:.6f} RFD:{rfd_bias_sq:.6f} RP:{rp_bias_sq:.6f} HS:{rp_hes_bias_sq:.6f}')
        if (fd_bias_sq < lower) or (fd_bias_sq > upper):
            print('⚠️ - BOUND NOT MET 🚫 ')
        if abs(fd_bias_sq - bias_g) < abs(rp_hes_bias_sq - bias_g) and abs(fd_bias_sq - bias_g) < abs(rp_bias_sq - bias_g):
            print('⭐️ FD win ⭐️')


    is_finite_idx = np.where(opt_bias > 0.)[0]
    fd_rel_err = np.abs(opt_bias[is_finite_idx]-fd_bias_vals[is_finite_idx])/opt_bias[is_finite_idx]
    rfd_rel_err = np.abs(opt_bias[is_finite_idx]-rfd_bias_vals[is_finite_idx])/opt_bias[is_finite_idx]
    rp_rel_err = np.abs(opt_bias[is_finite_idx]-rp_bias_vals[is_finite_idx])/opt_bias[is_finite_idx]
    hs_rel_err = np.abs(opt_bias[is_finite_idx]-hs_bias_vals[is_finite_idx])/opt_bias[is_finite_idx]

    fig,[ax,ax_b]=plt.subplots(nrows=2,ncols=1,dpi=150)
    ax.plot(all_gammas[is_finite_idx],fd_rel_err,label='FD')
    ax.plot(all_gammas[is_finite_idx],rfd_rel_err,label='RFD')
    ax.plot(all_gammas[is_finite_idx],rp_rel_err,label='Classical')
    ax.plot(all_gammas[is_finite_idx],hs_rel_err,label='Hessian')
    ax.legend()
    # ax.set_ylim(0.95,1.01)
    ax.set_yscale('log')
    ax.set_xscale('log',basex=2)
    ax.set_ylabel('Bias Squared (Relative Error)')
    ax.set_xlabel(r'$\gamma$')


    ax_b.plot(all_gammas[is_finite_idx],fd_bias_vals[is_finite_idx],label='FD')
    ax_b.plot(all_gammas[is_finite_idx],rfd_bias_vals[is_finite_idx],label='RFD')
    ax_b.plot(all_gammas[is_finite_idx],rp_bias_vals[is_finite_idx],label='Classical')
    ax_b.plot(all_gammas[is_finite_idx],hs_bias_vals[is_finite_idx],label='Hessian')
    ax_b.legend()
    # ax.set_ylim(0.95,1.01)
    ax_b.set_yscale('log')
    ax_b.set_xscale('log',basex=2)
    ax_b.set_ylabel('Bias Squared')# Relative Error')
    ax_b.set_xlabel(r'$\gamma$')
    plt.show()
    
def test_variance():
    n = 10**3
    d= 125
    eff_rank = int(floor(0.25*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank)
    # X,y,w0 = ds.shi_phillips_synthetic()
    # X /= np.linalg.norm(X,ord='fro')
    X,y,w0 = ds.mahoney_synthetic(noise_std=1.0)
    X /= np.linalg.norm(X,ord='fro')

    # Optimal bias
    mm = 64
    all_gammas = np.array([2**_ for _ in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]])
    H = X.T@X
    opt_var = np.zeros_like(all_gammas,dtype=float)
    lower_bound = np.zeros_like(opt_var)
    upper_bound = np.zeros_like(opt_var)
    fd_var_vals = np.zeros_like(opt_var)
    rfd_var_vals = np.zeros_like(opt_var)
    rp_var_vals = np.zeros_like(opt_var)
    hs_var_vals = np.zeros_like(opt_var)
    for i,g in enumerate(all_gammas):
        Hg = H + g*np.eye(d)
        #bias_g = (g**2)*np.linalg.norm(np.linalg.pinv(Hg)@w0)**2 #np.linalg.norm(x_opt - w0)**2 #
        var_g = np.linalg.norm(X@np.linalg.pinv(Hg),ord='fro')**2
        #print(bias_g)
        
        # FD Sketch
        fdr = FDRidge(fd_dim=mm,gamma=g)
        fdr.fit(X,y)
        fd_var = fdr.get_variance(X) 

        # RFD Sketch
        rfdr = FDRidge(fd_dim=mm,fd_mode='RFD',gamma=g)
        rfdr.fit(X,y)
        rfd_var = rfdr.get_variance(X) 
        
        # RP sketch
        rp = RPRidge(rp_dim=mm,gamma=g)
        rp.fit_classical(X,y)
        rp_var = rp.get_classical_variance(X)

        # Hessian Sketch
        rp.fit_hessian_sketch(X,y)
        hes_var = rp.get_hessian_sketch_variance(X)
        
        # Bounds 
        c = 1. - 1./(g*mm)
        if c < 0:
            continue
        upper = (1./c**2)*var_g
        lower = (c**2)*var_g
        opt_var[i] = var_g
        lower_bound[i] = lower 
        upper_bound[i] = upper
        fd_var_vals[i] = fd_var
        rfd_var_vals[i] = rfd_var
        rp_var_vals[i] = rp_var
        hs_var_vals[i] = hes_var
        print(f'γ:{g:.5f} c:{c:.5f} OPT:{var_g:.5f} Lower:{lower:.5f} Upper:{upper:.5f} FD:{fd_var:.5f} RFD:{rfd_var:.5f} RP:{rp_var:.5f} HS:{hes_var:.5f}')
        if (fd_var < lower) or (fd_var > upper):
            print('⚠️ - BOUND NOT MET 🚫 ')
        if abs(fd_var - var_g) < abs(rp_var - var_g) and abs(fd_var - var_g) < abs(hes_var - var_g) :
            print('⭐️ FD win ⭐️')

    is_finite_idx = np.where(opt_var > 0.)[0]
    fd_rel_err = np.abs(opt_var[is_finite_idx]-fd_var_vals[is_finite_idx])/opt_var[is_finite_idx]
    rfd_rel_err = np.abs(opt_var[is_finite_idx]-rfd_var_vals[is_finite_idx])/opt_var[is_finite_idx]
    rp_rel_err = np.abs(opt_var[is_finite_idx]-rp_var_vals[is_finite_idx])/opt_var[is_finite_idx]
    hs_rel_err = np.abs(opt_var[is_finite_idx]-hs_var_vals[is_finite_idx])/opt_var[is_finite_idx]

    fig,[ax,ax_v]=plt.subplots(nrows=2,ncols=1,dpi=150)
    ax.plot(all_gammas[is_finite_idx],fd_rel_err,label='FD')
    ax.plot(all_gammas[is_finite_idx],rfd_rel_err,label='RFD')
    ax.plot(all_gammas[is_finite_idx],rp_rel_err,label='Classical')
    ax.plot(all_gammas[is_finite_idx],hs_rel_err,label='Hessian')
    ax.legend()
    # ax.set_ylim(0.95,1.01)
    ax.set_yscale('log')
    ax.set_xscale('log',basex=2)
    ax.set_ylabel('Variance (Relative Error)')
    ax.set_xlabel(r'$\gamma$')


    ax_v.plot(all_gammas[is_finite_idx],fd_var_vals[is_finite_idx],label='FD')
    ax_v.plot(all_gammas[is_finite_idx],rfd_var_vals[is_finite_idx],label='RFD')
    ax_v.plot(all_gammas[is_finite_idx],rp_var_vals[is_finite_idx],label='Classical')
    ax_v.plot(all_gammas[is_finite_idx],hs_var_vals[is_finite_idx],label='Hessian')
    ax_v.legend()
    # ax.set_ylim(0.95,1.01)
    ax_v.set_yscale('log')
    ax_v.set_xscale('log',basex=2)
    ax_v.set_ylabel('Variance')# Relative Error')
    ax_v.set_xlabel(r'$\gamma$')
    plt.show()

def test_mse():
    n = 10**3
    d= 125
    eff_rank = int(floor(0.25*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank)
    # X,y,w0 = ds.shi_phillips_synthetic()
    # X /= np.linalg.norm(X,ord='fro')
    X,y,w0 = ds.mahoney_synthetic(noise_std=1.0)
    X /= np.linalg.norm(X,ord='fro')

    # Optimal bias
    mm = 64
    all_gammas = np.array([2**_ for _ in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]])
    H = X.T@X
    opt_mse = np.zeros_like(all_gammas,dtype=float)
    lower_bound = np.zeros_like(opt_mse)
    upper_bound = np.zeros_like(opt_mse)
    fd_mse_vals = np.zeros_like(opt_mse)
    rfd_mse_vals = np.zeros_like(opt_mse)
    rp_mse_vals = np.zeros_like(opt_mse)
    hs_mse_vals = np.zeros_like(opt_mse)
    for i,g in enumerate(all_gammas):
        Hg = H + g*np.eye(d)
        bias_g = (g**2)*np.linalg.norm(np.linalg.pinv(Hg)@w0)**2 #np.linalg.norm(x_opt - w0)**2 #
        var_g = np.linalg.norm(X@np.linalg.pinv(Hg),ord='fro')**2
        mse = var_g + bias_g
        
        # FD
        fdr = FDRidge(fd_dim=mm,gamma=g)
        fdr.fit(X,y)
        fd_bias = fdr.get_bias(X,w0) 
        fd_bias_sq = np.linalg.norm(fd_bias)**2 #np.linalg.norm(fdr.coef_ - w0)**2 #
        fd_var = fdr.get_variance(X) 
        fd_mse = fd_var + fd_bias_sq

        # FD
        rfdr = FDRidge(fd_dim=mm,fd_mode='RFD',gamma=g)
        rfdr.fit(X,y)
        rfd_bias = rfdr.get_bias(X,w0) 
        rfd_bias_sq = np.linalg.norm(rfd_bias)**2 
        rfd_var = rfdr.get_variance(X) 
        rfd_mse = rfd_var + rfd_bias_sq

        # RP sketch
        rp = RPRidge(rp_dim=mm,gamma=g)
        rp.fit_classical(X,y)
        rp_bias_sq = np.linalg.norm(rp.get_classical_bias(X,w0))**2
        rp_var = rp.get_classical_variance(X)
        rp_mse = rp_bias_sq + rp_var

        # Hessian Sketch
        rp.fit_hessian_sketch(X,y)
        rp_hes_bias_sq = np.linalg.norm(rp.get_hessian_sketch_bias(X,w0))**2
        hes_var = rp.get_hessian_sketch_variance(X)
        hes_mse = rp_hes_bias_sq + hes_var
        
        # Bounds 
        c = 1. - 1./(g*mm)
        if c < 0:
            continue
        upper = (1./c**2)*mse
        lower = (c**2)*mse
        opt_mse[i] = mse
        lower_bound[i] = lower 
        upper_bound[i] = upper
        fd_mse_vals[i] = fd_mse
        rfd_mse_vals[i] = rfd_mse
        rp_mse_vals[i] = rp_mse
        hs_mse_vals[i] = hes_mse
        print(f'γ:{g:.5f} c:{c:.5f} OPT:{mse:.5f} Lower:{lower:.5f} Upper:{upper:.5f} FD:{fd_mse:.5f} RFD:{rfd_mse:.5f} RP:{rp_mse:.5f} HS:{hes_mse:.5f}')
        if (fd_mse < lower) or (fd_mse > upper):
            print('⚠️ - BOUND NOT MET 🚫 ')
        if abs(fd_mse - mse) < abs(mse - hes_mse) :
            print('⭐️ FD win ⭐️')
        else:
            print(f'Error to HS: ', abs(fd_mse - mse), abs(hes_mse - mse))

    is_finite_idx = np.where(opt_mse > 0.)[0]
    fd_rel_err = np.abs(opt_mse[is_finite_idx]-fd_mse_vals[is_finite_idx])/opt_mse[is_finite_idx]
    rfd_rel_err = np.abs(opt_mse[is_finite_idx]-rfd_mse_vals[is_finite_idx])/opt_mse[is_finite_idx]
    rp_rel_err = np.abs(opt_mse[is_finite_idx]-rp_mse_vals[is_finite_idx])/opt_mse[is_finite_idx]
    hs_rel_err = np.abs(opt_mse[is_finite_idx]-hs_mse_vals[is_finite_idx])/opt_mse[is_finite_idx]

    fig,[ax,ax_mse]=plt.subplots(nrows=2,ncols=1,dpi=150)
    ax.plot(all_gammas[is_finite_idx],fd_rel_err,label='FD')
    ax.plot(all_gammas[is_finite_idx],rfd_rel_err,label='RFD')
    ax.plot(all_gammas[is_finite_idx],rp_rel_err,label='Classical')
    ax.plot(all_gammas[is_finite_idx],hs_rel_err,label='Hessian')
    ax.set_xscale('log',basex=2)
    ax.legend()
    # ax.set_ylim(0.95,1.01)
    ax.set_yscale('log')
    ax.set_ylabel('MSE Relative Error (Relative Error)')
    ax.set_xlabel(r'$\gamma$')

    ax_mse.plot(all_gammas[is_finite_idx],fd_mse_vals[is_finite_idx],label='FD')
    ax_mse.plot(all_gammas[is_finite_idx],rfd_mse_vals[is_finite_idx],label='RFD')
    #ax_mse.plot(all_gammas[is_finite_idx],rp_mse_vals[is_finite_idx],label='Classical')
    ax_mse.plot(all_gammas[is_finite_idx],hs_mse_vals[is_finite_idx],label='Hessian')
    ax_mse.legend()
    ax_mse.set_xlabel(r'$\gamma$')
    ax_mse.set_ylabel('MSE')
    ax_mse.set_yscale('log')
    ax_mse.set_xscale('log',basex=2)

    plt.show()
    


def main():
    # print('*'*10 , '\tOne shot\t', '*'*10 )
    # test_one_shot()
    print('*'*10 , '\tIterates\t', '*'*10 )
    test_iterates()
    # print('*'*10 , '\tBias term\t', '*'*10 )
    # test_bias()
    # print('*'*10 , '\tVariance\t', '*'*10 )
    # test_variance()
    # print('*'*10 , '\tMSE\t', '*'*10 )
    # test_mse()


if __name__ == '__main__':
    main()

