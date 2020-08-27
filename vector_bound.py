import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import floor
from pprint import PrettyPrinter
from scipy.optimize import curve_fit
from sklearn.random_projection import SparseRandomProjection
from frequentDirections import fit_fd
from utils import greater_than_thld, linear_fit, reciprocal_fit, gaussian_projection, sparse_projection
from data_factory import DataFactory

plt.style.use('classic')
np.set_printoptions(precision=4)
np.random.seed(100)



def vector_norm_bounds(data,sketch_start, sketch_end,nsamples=5,ax=None):
    """
    Runs and plots the vector norm bound experiment;
    a. generate nsamples of random vectors
    b. Estimates the norms from sketch_start to sketch_end
    c. plots result
    """  
    # Setup and ground truth
    _ , d = data.shape
    test_vecs = np.random.randn(d,nsamples)
    test_vecs = test_vecs / np.linalg.norm(test_vecs,axis=0)
    true_mat_vec = data@test_vecs
    true_mat_vec_norms = np.linalg.norm(true_mat_vec,axis=0,ord=2)**2

    # Initialise the error arrays
    sketch_dims = np.arange(start=sketch_start+1,stop=sketch_end,step=5)

    # Initialise FD arrays
    mean_percentage_decrease = np.zeros_like(sketch_dims,dtype=float)
    all_decreases = np.zeros((len(sketch_dims),nsamples))
    U,S,Vt = np.linalg.svd(data,full_matrices=False)
    X_k = U[:,:sketch_start]@(S[:sketch_start,None]*Vt[:sketch_start,:])
    delta_k = np.linalg.norm(data - X_k,ord='fro')**2
    error_bound = delta_k*np.ones_like(sketch_dims,dtype=float) 
    print('k = ', sketch_start)
    print(f'||A - Ak||_F^2 = {delta_k:.4f}')

    # Initialise random projection arrays
    rp_mean_percentage_decrease = np.zeros_like(mean_percentage_decrease)
    rp_all_decreases = np.zeros_like(all_decreases)




    #fig,ax = plt.subplots(dpi=100)
    for i,s in enumerate(sketch_dims):
        B, _ = fit_fd(data,proj_dim=s)
        sketch_mat_vec = B@test_vecs
        sketch_mat_vec_norms = np.linalg.norm(sketch_mat_vec,axis=0,ord=2)**2
        
        percentage_decrease = (true_mat_vec_norms - sketch_mat_vec_norms)/true_mat_vec_norms
        all_decreases[i,:] = percentage_decrease
        mean_percentage_decrease[i] = np.mean(percentage_decrease)
        error_bound[i] = np.max(delta_k/(s - sketch_start) * 1./ true_mat_vec_norms)
        print('*'*40)
        print('Sketch mat vec shape: ',sketch_mat_vec.shape)
        print(f'FD % decrease {np.mean(percentage_decrease):.4f}')
        print(f'Mean FD % decrease {mean_percentage_decrease[i]:.4f}')
        print(f'Error bound: {error_bound[i]:.4f}')

        # Evaluate the random projection performance:
        SA = sparse_projection(data,s,sparsity=5) #gaussian_projection(data,s)
        rp_mat_vec = SA@test_vecs
        rp_mat_vec_norms = np.linalg.norm(rp_mat_vec,axis=0,ord=2)**2
        rp_all_decreases[i,:] = (true_mat_vec_norms - rp_mat_vec_norms)/true_mat_vec_norms
        rp_mean_percentage_decrease[i] = np.median(np.abs(rp_all_decreases[i,:]))
        print(f'RP ABS % decrease {np.abs(rp_mean_percentage_decrease[i]):.4f}') 


    #print((sketch_dims/d).shape)
    #print(percentage_decrease.shape)
    #params = curve_fit(reciprocal_fit, sketch_dims/d, mean_percentage_decrease)
    #[a, b] = params[0]

    # Dict for plotting later
    res = {
        'k' : sketch_start/d,
        'd' : d,
        'Sketch size' : sketch_dims,
        'x' : sketch_dims/d,
        'Upper Bound' : error_bound,
        'Mean Distortion' : mean_percentage_decrease,
        #'Fitted Curve' : reciprocal_fit(sketch_dims/d,a,b),
        #'_curve_parameters' : [a,b],
        'Percentage Decrease' : all_decreases,
        'RP Approximation' : rp_all_decreases,
        'RP Mean Distortion' : rp_mean_percentage_decrease
    }

    return res

def plot_vector_norm_bounds(res,ax=None):
    if ax == None:
        _,ax = plt.subplots()
    d = res['d']
    sketch_dims = res['Sketch size']
    xx = res['x']
    percentage_decrease = res['Percentage Decrease']
    mean_distortion = res['Mean Distortion']
    upper_bound = res['Upper Bound']
    # fitted_curve = res['Fitted Curve']
    # a,b = res['_curve_parameters']

    # Get the gaussian results
    rp_approx = res['RP Approximation']
    rp_mean = res['RP Mean Distortion']


    # Scatter the estimates
    for i,s in enumerate(sketch_dims):
        ax.scatter( (s/d)*np.ones_like(percentage_decrease[i,:]) ,\
            percentage_decrease[i,:],\
            edgecolor=None,
            color='black',
            marker='.')

        ax.scatter( (s/d)*np.ones_like(rp_approx[i,:]) ,\
            np.abs(rp_approx[i,:]),\
            edgecolor='black',
            color='white',
            marker='o')
    # Plot the mean of estimates for FD
    ax.plot(xx, mean_distortion,linewidth=3.,label='Mean FD Distortion')

    # Plot the mean for RP
    ax.plot(xx, rp_mean,color='magenta',linewidth=3.,label='Mean RP distortion')

    # Plot the FD upper bound
    ax.plot(xx, upper_bound, label='Bound')


    # Plot the fitted curve
    # if b > 0:
    #     curve_str = r'${:.2f}/t + {:.2f}$'.format(a,b)
    # else:
    #      curve_str = r'${:.2f}/t - {:.2f}$'.format(a,-b) # the minus sign is a hack strip the -1 from numpy format without converting to str
    # ax.plot(xx, fitted_curve, label=curve_str)
    ax.plot([res['k'],res['k']], [0,1],color='red',linestyle=':',label=r'$k={:.2f}d$'.format(res['k']))


    ax.set_xlabel(r'$t = \mathrm{Sketch Dimension} / d$')
    ax.set_ylabel(r'$\frac{\|Ax\|^2 - \|Bx\|^2}{\|Ax^2\|}$')
    # ax.set_ylim(np.min([mean_distortion, fitted_curve])-0.05,1.05)
    # ax.set_xlim(sketch_dims[0]/d-0.5,1.05)
    ax.grid(True)
    #ax.set_ylim(-0.05,1)

    # Log scale axis setting
    ax.set_yscale('log')
    #ax.set_xscale('log')
    #ax.set_xlim(1E-1-0.05,1.+0.05)
    ax.set_ylim(1E-4)

    
    ax.legend()


def main():
    # pp = PrettyPrinter(indent=4)
    # data = np.load('data/superconductor.npy')
    # X,_y = data[:,:-1], data[:,-1]
    # n = 10000
    # d = 500
    # tail_strength = 0.1
    # R_lr = floor(0.1*d + 0.5)
    # X = DataFactory(n,d).fetch_low_rank_matrix(R_lr,tail_strength)#fetch_superconductor()
    #X,_ = DataFactory().fetch_superconductor()
    X = DataFactory().fetch_newsgroups(ncols=500)
    [n,d] = X.shape

    # 1a. Get the optimal SVD
    U,S,Vt = np.linalg.svd(X,full_matrices=False)
    # fig_spec, ax_spec = plt.subplots(dpi=100)
    # ax_spec.plot(range(d),S,label='Spectrum')


    # 1b. Evaluate rank estimation
    frobenius_estimation = np.zeros(d)
    for _,k in enumerate(range(1,d+1)):
        Xk = U[:,:k]@(S[:k,None]*Vt[:k,:])
        frobenius_estimation[_] = np.linalg.norm(X - Xk,ord='fro')**2
    frobenius_estimation /= np.linalg.norm(X,ord='fro')**2

    # for i in range(1,d+1):
    #     print(f'k = {i}, error = {(1. - frobenius_estimation[i]):.4f}')

    k_75 = greater_than_thld(1.0 - frobenius_estimation,0.75)
    k_90 = greater_than_thld(1.0 - frobenius_estimation,0.90)
    k_99 = greater_than_thld(1.0 - frobenius_estimation,0.99)
    print('75% Thld: ', k_75)
    print('90% Thld: ', k_90)
    print('99% Thld: ', k_99)

    fig,axarr = plt.subplots(nrows=1,ncols=3,dpi=100)

    rank_k = [k_75, k_90, k_99]
    for i,ax in enumerate(axarr):
        k = rank_k[i]
        res = vector_norm_bounds(X,int(k),d//2,10)
        print('*'*40)
        print('k = ', k)
        plot_vector_norm_bounds(res, ax)
    plt.show()
    

    
    # fig, axarr = plt.subplots(3, 3, gridspec_kw = {'wspace':0, 'hspace':0})

    # for i, ax in enumerate(fig.axes):
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    # plt.show()


    # 2 Make the FD sketch and visualise spectrum
    # sketch_size = 40
    # B, alpha = fit_fd(X,proj_dim=sketch_size)
    # Uf, Sf, Vtf = np.linalg.svd(B,full_matrices=False)
    # Urf, Srf, Vtrf = np.linalg.svd(B.T@B + alpha*np.eye(d),full_matrices=False)
    # print('Frequent Directions distortion')
    # vector_norm_error(X,B)

    # SX = (1./np.sqrt(sketch_size))*np.random.randn(sketch_size,n) @ X
    # print('Gaussin Projection Distortion')
    # vector_norm_error(X,SX)
    # 3. Visualise the spectra
    # fig,ax = plt.subplots(dpi=100)
    # ax.plot(range(d),S,label='True Singular Values')
    # ax.plot(range(sketch_size),Sf,label='FD Singular Values')
    # ax.plot(range(d),np.sqrt(Srf),label='RFD Singular Values')
    # ax.fill_between(range(k_90), 0, S[:k_90],label=f'90%%@{k_90}', alpha=0.5)
    # ax.fill_between(range(k_99), 0, S[:k_99],label=f'99%%@{k_99}', alpha=0.25)
    # ax.set_xlabel('Index')
    # ax.set_ylabel('Singular Value')
    # ax.legend(loc='upper right')
    # ax.set_xlim(-0.1)
    # ax.set_ylim(-0.1)
    # #ax.set_yscale('log')
    # plt.show()

if __name__ == '__main__':
    main()


# def vector_norm_error(data,sketch,nsamples=10):
    # '''
    # Establishes the vector norm raltionship from the FD paper by
    # sampling unit length gaussian vectors and gettting the norm.
    # We generate a matrix whose columns are standard normal vectors 
    # which are then normalised to unit length.
    # '''
    # _,d = data.shape
    # test_vecs = np.random.randn(d,nsamples)
    # test_vecs = test_vecs / np.linalg.norm(test_vecs,axis=0)
    # true_mat_vec = data@test_vecs
    # sketch_mat_vec = sketch@test_vecs
    # true_mat_vec_norms = np.linalg.norm(true_mat_vec,axis=0,ord=2)**2
    # sketch_mat_vec_norms = np.linalg.norm(sketch_mat_vec,axis=0,ord=2)**2
    # print(np.c_[true_mat_vec_norms,sketch_mat_vec_norms, 
    #             np.abs(true_mat_vec_norms - sketch_mat_vec_norms)/true_mat_vec_norms])
    # relative_error = np.abs(true_mat_vec_norms - sketch_mat_vec_norms)/true_mat_vec_norms
    # mean_relative_error = np.sum(relative_error)/nsamples
    # print(f'Mean distortion {mean_relative_error:.4f}')  

# def covariance_error(data,sketch_start, sketch_end):
    # """
    # Experiment to test the performance of the covariance error perspective.
    # This is the 'worst case' plot as 
    # """
    # _ , d = data.shape
    # cov = data.T @ data

    # # Initialise the error arrays
    # sketch_dims = np.arange(start=sketch_start+1,stop=sketch_end,step=2)
    # fd_cov_est_error = np.zeros_like(sketch_dims,dtype=float)
    # rfd_cov_est_error = np.zeros_like(sketch_dims,dtype=float)
    # U,S,Vt = np.linalg.svd(data,full_matrices=False)
    # X_k = U[:,:sketch_start]@(S[:sketch_start,None]*Vt[:sketch_start,:])
    # delta_k = np.linalg.norm(data - X_k,ord='fro')**2
    # upper_bound = delta_k*np.ones_like(sketch_dims,dtype=float) #nsamples as we take mean

    # for i,s in enumerate(sketch_dims):
    #     B,a = fit_fd(data, s)
    #     H = B.T@B
    #     fd_cov_est_error[i] = np.linalg.norm(cov - H,ord=2)
    #     rfd_cov_est_error[i] = np.linalg.norm(cov - (H + a*np.eye(d)),ord=2)
    #     upper_bound[i] /= (s - sketch_start)
    # fig,ax = plt.subplots(dpi=100)
    # ax.plot(sketch_dims/d, fd_cov_est_error/delta_k, label='FD')
    # ax.plot(sketch_dims/d, rfd_cov_est_error/delta_k, label='RFD')
    # ax.plot(sketch_dims/d, upper_bound/delta_k,label='Bound')
    # ax.set_xlabel('Sketch Dimension / d')
    # ax.set_ylabel('Covariance Error')
    # ax.legend()
    # plt.show()