import sys
from data_factory import DataFactory
from frequent_directions import FastFrequentDirections, RobustFrequentDirections
from frequent_directions_regression import FDRidge
from random_projection_regression import RPRidge
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.datasets import make_low_rank_matrix
from plot_config import fd_params, rfd_params, rp_params, hs_params
from sklearn.datasets import make_regression
import numpy as np
from math import floor
import matplotlib.pyplot as plt

def mse_experiment():
    n = 2**10
    d = 2**9
    eff_rank = int(floor(0.2*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank,tail_strength=0.125)
    X,y,w0 = ds.shi_phillips_synthetic()
    #X /= np.linalg.norm(X,ord='fro')
    #X,y,w0 = ds.mahoney_synthetic(noise_std=1.0)
    #X /= np.linalg.norm(X,ord='fro')
    #X,y,w0 = make_regression(n_samples=n,n_features=d,n_informative=eff_rank,effective_rank=eff_rank,tail_strength=0.125,noise=10.0,coef=True)
    # X,y,w0 = ds.chowdury_synthetic()
    # _,s,_ = np.linalg.svd(X,full_matrices=False)
    

    # Optimal bias
    mm = 256
    all_gammas = np.array([2**_ for _ in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]])
    H = X.T@X

    # Initialise the bound arrays
    opt_mse = np.zeros_like(all_gammas,dtype=float)
    opt = {
        'Bias'                  : np.zeros_like(opt_mse),
        'Variance'              : np.zeros_like(opt_mse),
        'MSE'                   : np.zeros_like(opt_mse),
        'Bias: Lower Bound'     : np.zeros_like(opt_mse),
        'Bias: Upper Bound'     : np.zeros_like(opt_mse),
        'Variance: Lower Bound' : np.zeros_like(opt_mse),
        'Variance: Upper Bound' : np.zeros_like(opt_mse),
        'MSE: Lower Bound'      : np.zeros_like(opt_mse),
        'MSE: Upper Bound'      : np.zeros_like(opt_mse)
    }

    # Initialise measurement arrays
    fd = {
        'Bias'     : np.zeros_like(opt_mse),
        'Variance' : np.zeros_like(opt_mse),
        'MSE'      : np.zeros_like(opt_mse)
    }
    rfd = {
        'Bias'     : np.zeros_like(opt_mse),
        'Variance' : np.zeros_like(opt_mse),
        'MSE'      : np.zeros_like(opt_mse)
    }
    cl = {
        'Bias'     : np.zeros_like(opt_mse),
        'Variance' : np.zeros_like(opt_mse),
        'MSE'      : np.zeros_like(opt_mse)
    }
    hs = {
        'Bias'     : np.zeros_like(opt_mse),
        'Variance' : np.zeros_like(opt_mse),
        'MSE'      : np.zeros_like(opt_mse)
    }

    # Execute the sketches

    for i,g in enumerate(all_gammas):
        print(f'Testing γ = {g:.4f}')
        Hg = H + g*np.eye(d)
        bias_g = (g**2)*np.linalg.norm(np.linalg.pinv(Hg)@w0)**2 #np.linalg.norm(x_opt - w0)**2 #
        var_g = np.linalg.norm(X@np.linalg.pinv(Hg),ord='fro')**2
        mse = var_g + bias_g
        
        # FD
        print('FD Sketching')
        fdr = FDRidge(fd_dim=mm,gamma=g)
        fdr.fit(X,y)
        fd_bias = fdr.get_bias(X,w0) 
        fd_bias_sq = np.linalg.norm(fd_bias)**2 #np.linalg.norm(fdr.coef_ - w0)**2 #
        fd_var = fdr.get_variance(X) 
        fd_mse = fd_var + fd_bias_sq
        print('FD complete')

        # FD
        print('RFD Sketching')
        rfdr = FDRidge(fd_dim=mm,fd_mode='RFD',gamma=g)
        rfdr.fit(X,y)
        rfd_bias = rfdr.get_bias(X,w0) 
        rfd_bias_sq = np.linalg.norm(rfd_bias)**2 
        rfd_var = rfdr.get_variance(X) 
        rfd_mse = rfd_var + rfd_bias_sq
        print('RFD complete')

        # RP sketch
        print('RP Sketching')
        rp = RPRidge(rp_dim=mm,gamma=g)
        rp.fit_classical(X,y)
        rp_bias_sq = np.linalg.norm(rp.get_classical_bias(X,w0))**2
        rp_var = rp.get_classical_variance(X)
        rp_mse = rp_bias_sq + rp_var
        print('RP complete')

        # Hessian Sketch
        print('HS Sketching')
        rp.fit_hessian_sketch(X,y)
        hes_bias_sq = np.linalg.norm(rp.get_hessian_sketch_bias(X,w0))**2
        hes_var = rp.get_hessian_sketch_variance(X)
        hes_mse = hes_bias_sq + hes_var
        print('HS complete')
        
        # Bounds 
        c = 1. - 1./(g*mm)
        if c <= 0:
            continue
        upper = (1./c**2)*mse
        lower = (c**2)*mse

        # Optimal values
        opt['Bias'][i] = bias_g
        opt['Variance'][i] = var_g
        opt['MSE'][i] = mse
        opt['Bias: Lower Bound'][i] = (c**2)*bias_g
        opt['Bias: Upper Bound'][i] = (1./c**2)*bias_g
        opt['Variance: Lower Bound'][i] = (c**2)*var_g
        opt['Variance: Upper Bound'][i] = (1./c**2)*var_g
        opt['MSE: Lower Bound'][i] = (c**2)*mse
        opt['MSE: Upper Bound'][i] = (1./c**2)*bias_g

        # Measurements: FD
        fd['Bias'][i]     =  fd_bias_sq
        fd['Variance'][i] =  fd_var
        fd['MSE'][i]      =  fd_mse 

        # Measurements RFD
        rfd['Bias'][i]     =  rfd_bias_sq
        rfd['Variance'][i] =  rfd_var
        rfd['MSE'][i]      =  rfd_mse 

        # Measurements RP
        cl['Bias'][i]     =  rp_bias_sq
        cl['Variance'][i] =  rp_var
        cl['MSE'][i]      =  rp_mse 

        # Measurements HS
        hs['Bias'][i]     =  hes_bias_sq
        hs['Variance'][i] =  hes_var
        hs['MSE'][i]      =  hes_mse 
  

        print(f'γ:{g:.5f} c:{c:.5f} OPT:{mse:.5f} Lower:{lower:.5f} Upper:{upper:.5f} FD:{fd_mse:.5f} RFD:{rfd_mse:.5f} RP:{rp_mse:.5f} HS:{hes_mse:.5f}')
        if (fd_mse < lower) or (fd_mse > upper):
            print('⚠️ - BOUND NOT MET 🚫 ')
        if abs(fd_mse - mse) < abs(mse - hes_mse) :
            print('⭐️ FD win ⭐️')
        else:
            print(f'Error to HS: ', abs(fd_mse - mse), abs(hes_mse - mse))

    # Make the plots 
    fig, ax = plt.subplots(nrows=2,ncols=3,gridspec_kw = {'wspace':0.125, 'hspace':0.0},figsize=(16,8))
    # Row 1 is the relative error plots
    # Bias-Variance-MSE
    ax[0,0].title.set_text("Bias Squared")
    ax[0,1].title.set_text("Variance")
    ax[0,2].title.set_text("MSE")
    ax[0,0].set_ylabel("Relative Error")

    for ax_i,measure in enumerate(['Bias', 'Variance', 'MSE']):
        finite_ids = np.where(opt[measure] > 0.)[0]
        gammas_finite = all_gammas[finite_ids]
        fd_rel_err = np.abs(opt[measure][finite_ids]- fd[measure][finite_ids])/ opt[measure][finite_ids]
        rfd_rel_err = np.abs(opt[measure][finite_ids]- rfd[measure][finite_ids])/ opt[measure][finite_ids]
        rp_rel_err = np.abs(opt[measure][finite_ids]- cl[measure][finite_ids])/ opt[measure][finite_ids]
        hs_rel_err = np.abs(opt[measure][finite_ids]- hs[measure][finite_ids])/ opt[measure][finite_ids]

        ax[0,ax_i].plot(gammas_finite,fd_rel_err,label='FD',**fd_params)
        ax[0,ax_i].plot(gammas_finite,rfd_rel_err,label='RFD',**rfd_params)
        ax[0,ax_i].plot(gammas_finite,rp_rel_err,label='Classical', **rp_params)
        ax[0,ax_i].plot(gammas_finite,hs_rel_err,label='Hessian', **hs_params)
        ax[0,ax_i].set_yscale('log', basey=10)
        ax[0,ax_i].set_xscale('log', basex=10)
        #ax[0,ax_i].set_yticklabels(gammas_finite,rotation=45)
   

    # Row 2 is the absolute value plots
    ax[1,0].set_ylabel("Absolute Values")
    for ax_i,measure in enumerate(['Bias', 'Variance', 'MSE']):
        finite_ids = np.where(opt[measure] > 0.)[0]
        gammas_finite = all_gammas[finite_ids]
        #opt_m = opt[measure][finite_ids]
        #print(opt_m)
        fd_m = fd[measure][finite_ids]
        rfd_m = rfd[measure][finite_ids]
        cl_m = cl[measure][finite_ids]
        hs_m = hs[measure][finite_ids]
        # if measure == 'Bias':
        #     print(np.c_[fd_m,rfd_m,cl_m,hs_m])

        #ax[1,ax_i].plot(gammas_finite,opt_m,label='OPT',color='cyan',marker='+',markersize=10)

        ax[1,ax_i].plot(gammas_finite,fd_m,label='FD',**fd_params)
        ax[1,ax_i].plot(gammas_finite,rfd_m,label='RFD',**rfd_params)
        ax[1,ax_i].plot(gammas_finite,cl_m,label='Classical', **rp_params)
        if measure != 'Bias':
            ax[1,ax_i].plot(gammas_finite,hs_m,label='Hessian', **hs_params)
        if ax_i > 0: # Override log scale for ax[1,0] as similar magnitude
            ax[1,ax_i].set_yscale('log', basey=10)
        #ax[1,ax_i].set_yscale('log', basey=10)
        ax[1,ax_i].set_xscale('log', basex=10)
        ax[1,ax_i].set_xlabel(r'$\gamma$')
    ax[1,2].legend(loc='upper right') # Only plot legend in one of the plots
    #fig.tight_layout() 
    fig.savefig('figures/bias-variance-tradeoff.pdf', 
                dpi=300, bbox_inches='tight')
    #             facecolor='w', edgecolor='w')
    # bbox_inches='tight')
    



 

    plt.show()
    

def main():
    mse_experiment()

if __name__ == '__main__':
    main()