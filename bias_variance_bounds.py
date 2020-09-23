from data_factory import DataFactory
from frequent_directions_regression import FDRidge
from random_projection_regression import RPRidge
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import plot_config 

FIGPATH = 'figures/'

def mse_performance():
    '''
    Experimental Setup:
    - 
    '''
    n = 2**12
    d = 2**10
    eff_rank = int(floor(0.1*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank,random_seed=100)
    #X,y,w0 = ds.mahoney_synthetic(noise_std=0.1)
    X,y,w0 = ds.shi_phillips_synthetic()
    #X /= np.linalg.norm(X,ord='fro')

    # Optimal bias
    mm = 2**8
    all_gammas = np.array([2**_ for _ in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]])
    H = X.T@X
    opt_bias = np.zeros_like(all_gammas,dtype=float)
    opt_var = np.zeros_like(all_gammas,dtype=float)
    opt_mse = np.zeros_like(all_gammas,dtype=float)

    # Bound arrays
    bias_lower_bound = np.zeros_like(opt_bias)
    bias_upper_bound = np.zeros_like(opt_bias)
    var_upper_bound = np.zeros_like(opt_bias)
    var_lower_bound = np.zeros_like(opt_bias)
    mse_upper_bound = np.zeros_like(opt_bias)
    mse_lower_bound = np.zeros_like(opt_bias)

    # Measurement arrays
    fd_bias_vals = np.zeros_like(opt_bias)
    rfd_bias_vals = np.zeros_like(opt_bias)
    fd_var_vals = np.zeros_like(opt_var)
    rfd_var_vals = np.zeros_like(opt_var)
    fd_mse_vals = np.zeros_like(opt_mse)
    rfd_mse_vals = np.zeros_like(opt_mse)

    for i,g in enumerate(all_gammas):
        Hg = H + g*np.eye(d)
        bias_g = (g**2)*np.linalg.norm(np.linalg.pinv(Hg)@w0)**2 
        var_g = np.linalg.norm(X@np.linalg.pinv(Hg),ord='fro')**2
        mse = var_g + bias_g
        
        # FD Sketch
        fdr = FDRidge(fd_dim=mm,gamma=g)
        fdr.fit(X,y)
        fd_bias = fdr.get_bias(X,w0) 
        fd_bias_sq = np.linalg.norm(fd_bias)**2 
        fd_var = fdr.get_variance(X) 
        fd_mse = fd_var + fd_bias_sq

        # RFD Sketch
        rfdr = FDRidge(fd_dim=mm,fd_mode='RFD',gamma=g)
        rfdr.fit(X,y)
        rfd_bias = rfdr.get_bias(X,w0) 
        rfd_bias_sq = np.linalg.norm(rfd_bias)**2 
        rfd_var = rfdr.get_variance(X) 
        rfd_mse = rfd_var + rfd_bias_sq

        
        # Bounds 
        #c = 1. - 1./(g*mm)
        theta = (1/(g*mm))*(2 - 1/(g*mm))
        if theta <= 0:
            continue
        # Set up the bound arrays
        approx_factor = 1. - theta
        b_upper = (1./approx_factor)*bias_g
        b_lower = approx_factor*bias_g
        v_upper = (1./approx_factor)*var_g
        v_lower = approx_factor*var_g
        m_upper = (1./approx_factor)*mse
        m_lower = approx_factor*mse

        opt_bias[i] = bias_g
        opt_var[i] = var_g
        bias_lower_bound[i] = b_lower 
        bias_upper_bound[i] = b_upper
        var_lower_bound[i] = v_lower
        var_upper_bound[i] = v_upper
        mse_lower_bound[i] = m_lower
        mse_upper_bound[i] = m_upper
        

        # Measurements
        fd_bias_vals[i] = fd_bias_sq
        fd_var_vals[i] = fd_var
        fd_mse_vals[i] = fd_mse
        rfd_bias_vals[i] = rfd_bias_sq
        rfd_var_vals[i] = rfd_var
        rfd_mse_vals[i] = rfd_mse

        print(f'γ:{g:.6f} theta:{theta:.5f} OPT:{mse:.6f} Lower:{m_lower:.6f} Upper:{m_upper:.6f} FD:{fd_mse:.6f} RFD:{rfd_mse:.6f}')
        if (fd_bias_sq < b_lower) or (fd_bias_sq > b_upper):
            print('⚠️ - BOUND NOT MET 🚫 ')


    is_finite_idx = np.where(opt_bias > 0.)[0]
    fd_rel_err = np.abs(opt_bias[is_finite_idx]-fd_bias_vals[is_finite_idx])/opt_bias[is_finite_idx]
    rfd_rel_err = np.abs(opt_bias[is_finite_idx]-rfd_bias_vals[is_finite_idx])/opt_bias[is_finite_idx]

    plot_bias_bounds(all_gammas[is_finite_idx],bias_lower_bound[is_finite_idx],bias_upper_bound[is_finite_idx],fd_bias_vals[is_finite_idx])
    plot_variance_bounds(all_gammas[is_finite_idx],var_lower_bound[is_finite_idx],var_upper_bound[is_finite_idx],fd_var_vals[is_finite_idx])
    plot_mse_bounds(all_gammas,mse_lower_bound,mse_upper_bound,fd_mse_vals)
    plt.show()


def plot_bias_bounds(gammas,lower,upper,bias):
    '''
    Plots the bounds for bias terms
    '''
    bias_med = np.median(bias)
    bias_std = np.std(bias)
    fig,ax = plt.subplots(dpi=125,figsize=(5,3))
    ax.plot(gammas,lower,linestyle=':',linewidth=1.5,color='red',label='Bounds')
    ax.plot(gammas,upper,linestyle=':',linewidth=1.5,color='red')
    ax.plot(gammas,bias,linestyle='-',linewidth=1.5,color='black',label='FD')
    ax.set_ylim(bias_med-2.5*bias_std,bias_med+2.5*bias_std)
    ax.set_ylabel('Bias (Squared Norm)')
    ax.set_xscale('log',basex=2)
    ax.set_xlabel(r'$\gamma$')
    ax.legend()
    fname = FIGPATH+'bias_bound.pdf'
    fig.savefig(fname,dpi=125, facecolor='w', edgecolor='w',
        bbox_inches=None)


def plot_variance_bounds(gammas,lower,upper,variance):
    '''
    Plots the bounds for variance terms
    '''
    is_finite_idx = np.where(np.isfinite(upper))[0]
    lower = lower[is_finite_idx]
    upper = upper[is_finite_idx]
    variance = variance[is_finite_idx]
    gammas = gammas[is_finite_idx]

    # var_med = np.median(variance)
    # var_std = np.std(variance)
    fig,ax = plt.subplots(dpi=125,figsize=(5,3))
    ax.plot(gammas,lower,linestyle=':',linewidth=1.5,color='red',label='Bounds')
    ax.plot(gammas,upper,linestyle=':',linewidth=1.5,color='red')
    ax.plot(gammas,variance,linestyle='-',linewidth=1.5,color='black',label='FD')
    #ax.set_ylim(0.,10.)
    ax.set_ylabel('Variance')
    ax.set_xlabel(r'$\gamma$')
    #ax.set_ylim(lower.min(),upper.max())
    ax.set_yscale('log',basey=2)
    ax.set_xscale('log',basex=2)
    #ax.legend()
    fname = FIGPATH+'variance_bound.pdf'
    fig.savefig(fname,dpi=125, facecolor='w', edgecolor='w',
        bbox_inches=None)

def plot_mse_bounds(gammas,lower,upper,mse):
    '''
    Plots the bounds for variance terms
    '''
    is_finite_idx = np.where(np.isfinite(upper))[0]
    lower = lower[is_finite_idx]
    upper = upper[is_finite_idx]
    mse = mse[is_finite_idx]
    gammas = gammas[is_finite_idx]

    mse_med = np.median(mse)
    mse_std = np.std(mse)
    fig,ax = plt.subplots(dpi=125,figsize=(5,3))
    ax.plot(gammas,lower,linestyle=':',linewidth=1.5,color='red',label='Bounds')
    ax.plot(gammas,upper,linestyle=':',linewidth=1.5,color='red')
    ax.plot(gammas,mse,linestyle='-',linewidth=1.5,color='black',label='FD')
    #ax.set_ylim(0.,10.)
    ax.set_ylabel('MSE')
    ax.set_xlabel(r'$\gamma$')
    #ax.set_ylim(lower.min(),upper.max())
    ax.set_yscale('log',basey=2)
    ax.set_xscale('log',basex=2)
    #ax.legend()
    fname = FIGPATH+'mse_bound.pdf'
    fig.savefig(fname,dpi=125, facecolor='w', edgecolor='w',
        bbox_inches=None)
    


def main():
    mse_performance()

if __name__ == '__main__':
    main()