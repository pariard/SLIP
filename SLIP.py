#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the paper

"SLIP: Learning to predict in unknown dynamical systems with long-term memory"
Authors: Paria Rashidinejad, Jiantao Jiao, Stuart Russell

"""
##### Imports #####
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Garamond']})
rc('text', usetex=True)
rc('legend', fontsize=25)
rc('font', size=30)    
plt.rcParams['axes.labelsize'] = 25 
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize']= 25
plt.rcParams['ytick.labelsize']= 25

import numpy as np
from scipy.stats import norm
from numpy.linalg import matrix_rank, matrix_power, inv, eig, svd, eigvals
from scipy.linalg import pinvh, eigh, svdvals
matrix_norm = np.linalg.norm
from sklearn.linear_model import Ridge
import time
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


#%% Functions: Use to check reachability and observability conditions to ensure existence of stationary Kalman filter
def is_reachable(A,C):
    """
    Inputs: LDS matrices A and C
    Output: returns whether (A,C) reachable
    """
    (m,d) = C.shape
    M = C.T
    for i in range(1,m):
        M = np.concatenate((M, np.matmul(matrix_power(A,i),C.T)))
    r = matrix_rank(M)
    if r == min(m,d): # M is full rank
        return True
    else:
        return False
    
def is_observable(A,C):
    """
    Inputs: LDS matrices A and C
    Output: returns whether (A,C) observable
    """
    return is_reachable(A.T,C.T)

def generate_random_LDS(n,d,m, A_spectral_radius, max_entry = 1):
    """
    Inputs:
        n = input dimension
        m = output dimension
        d = hidden state dimension
        A_spectral_radius = spectral radius of A
        max_entry = maximum entry for parameter matrices
    Outputs:
        A, B, C, D, Q, R
    """
    B = np.random.uniform(-max_entry, max_entry, size = [d,n]) 
    D = np.random.uniform(-max_entry, max_entry, size = [m,n]) 
    J = np.random.uniform(-max_entry, max_entry, size = [m,m]) # R = JJ^T
    R = np.matmul(J,J.T) # observation noise covariance
    
    # generate matrices A, C, Q = UU^T so that the system is reachable and observable
    stationarity_condition = False
    max_count = 100
    count = 0
    while(stationarity_condition == False and count < max_count):
        A = np.random.uniform(-max_entry, max_entry, size = [d,d])   
        A = A/matrix_norm(A,2)*A_spectral_radius # set the specified spectral radius for A
        C = np.random.uniform(-max_entry, max_entry, size = [m,d])
        U = np.random.uniform(-max_entry, max_entry, size = [d,d])
        Q = np.matmul(U,U.T)     
        observability = is_observable(A.T,U.T)
        reachability = is_reachable(A,C)
        if observability and reachability:
            stationarity_condition = True
        else:
            count += 1 
    
    if count == 100:
        if not reachability:
            raise ValueError('(A,C) is not reachable.')
        if not observability:
            raise ValueError('(A.T,U.T) is not observable.')
            
    return A, B, C, D, Q, R

#%% LDS Settings: Change these to test different systems
draw_plots = True


### LDS dimensions
n = 1 # input dimension
d = 4 # hidden state dimension
m = 1 # output dimension


### System parameters
max_entry = 1 # max abs entry for B, C, D
A_spectral_radius = 0.9 # set to <= 1 to ensure marginal stability

A, B, C, D, Q, R = generate_random_LDS(n,d,m, A_spectral_radius, max_entry)

### Time horizon
T = 1000

### Total sampled trajectories 
num_iterations = 10 

filter_count = 10 # hyperparameter for the number of filters
alpha_reg = 10**(-4) # regularization parameter

### Define the variables
x = np.zeros([n,T]) # inputs
h = np.zeros([d,T]) # hidden states
y = np.zeros([m,T]) # observations
eta = np.zeros([d,T]) # process noise
zeta = np.zeros([m,T]) # observation noise



#%% spectral filters and regularization
       
# Hankel matrix given in Eq (8) in paper        
H = np.zeros([T,T]) 
for i in range(T):
    for j in range(T):
        H[i,j]= 0.5*((-1)**(i+j)+1)/(i+j+1)
        
eigen_values, eigen_vectors = eigh(H,eigvals = (T-filter_count,T-1))
sigma = eigen_values[-1:-1-filter_count:-1] # eigenvalues to normalize
phi = eigen_vectors[:,-1:-1-filter_count:-1] # spectral filters

# reverse the eigenvectors
for filter_num in range(filter_count):
    indx = np.argmax(abs(phi[:,filter_num]))
    if phi[indx,filter_num] < 0:
        phi[:,filter_num] = - phi[:,filter_num]   


#%%
prediction_error = np.zeros([T,num_iterations]) # computes ||y_t - hat{m}_t||^2
difference_error = np.zeros([T,num_iterations]) # computes ||m_t - hat{m}_t||^2: difference between Kalman in hindsight predictions and SLIP prediction

for iteration in range(num_iterations):
    t_start = time.time()
    
    # (I) Sample/specify inputs
    #Rx = 0.01 # max entry for inputs used for uniform inputs
    x = norm.rvs(0,1,size = [n,T])
    x[:,0] = 0 # initial input is zero
    
    # (II) Sample Gaussian noise processes
    eta = np.random.multivariate_normal(np.zeros(d), Q, size = T).T
    zeta = np.random.multivariate_normal(np.zeros(m), R, size = T).T
    
    # (III) Initialize Kalman filter variables
    mu = np.zeros([d,T]) # hidden state estimator
    Sigma = np.zeros([d,d,T]) # hidden state covariance matrix
    mu_p = np.zeros([d,T]) # hidden state predictor
    Sigma_p = np.zeros([d,d,T]) # hidden state prediction covariance matrix
    m_kf = np.zeros([m,T]) # next observation prediction by Kalman filter in hindsight
    V_kf = np.zeros([m,m,T]) # next observation prediction covariance matrix by Kalman filter in hindsight
    K_kf = np.zeros([d,m,T]) # Kalman gain
    G_kf = np.zeros([d,d,T]) # G=A(I-KC)
    e = np.zeros_like(m_kf)  # innovations
    
    
    # (IV) SLIP variables and parameters
    
    # variables
    m_sf = np.zeros([m,T]) # next observation prediction by SLIP
    x0_sf = np.zeros([n,T]) # current input (feature) x0_sf = x_t 
    Y_sf = np.zeros([m,filter_count,T]) # (observation features) Y_tilde_sf(:,h,t) = sigma_h^{0.25} y[:,1:t] phi[t:1]
    X_sf = np.zeros([n,filter_count,T]) # (input features) X_tilde_sf(:,h,t) = sigma_h^{0.25} x[:,1:t] phi[t:1]
    num_features = (n+m)*filter_count + n
    Features = np.zeros([num_features ,T])

    # parameters 
    Theta = np.zeros([m,num_features,T])
    Theta[:,:,0] = norm.rvs(0,1,size = Theta[:,:,0].shape) # initialize at step 0
    
    for t in range(1,T):

        # dynamics
        h[:,t] = np.matmul(A,h[:,t-1]) + np.matmul(B,x[:,t-1]) + eta[:,t-1]
        y[:,t] = np.matmul(C,h[:,t]) + np.matmul(D,x[:,t]) + zeta[:,t]
        
        #### ______________Kalman filter in hindsight updates___________
        mu_p[:,t] = np.matmul(A, mu[:,t-1])+ np.matmul(B,x[:,t-1])
        Sigma_p[:,:,t] = np.matmul(np.matmul(A,Sigma[:,:,t-1]),A.T) + Q
        m_kf[:,t] = C.dot(mu_p[:,t]) + D.dot(x[:,t])
        V_kf[:,:,t] = np.matmul(np.matmul(C,Sigma_p[:,:,t]), C.T) + R
        K_kf[:,:,t] = np.matmul(np.matmul(Sigma_p[:,:,t],C.T), inv(V_kf[:,:,t]))
        gain = np.eye(d)-np.matmul(K_kf[:,:,t],C)
        G_kf[:,:,t] = np.matmul(A,gain)
        mu[:,t] = gain.dot(mu_p[:,t]) + np.matmul(K_kf[:,:,t],y[:,t] - D.dot(x[:,t]))
        Sigma[:,:,t] = np.matmul(gain,Sigma_p[:,:,t])
        e[:,t] = y[:,t] - m_kf[:,t] 
        
        ####_______________SLIP updates_________________
        
        # SLIP predictions, using parameters from t-1        
        Features[0:n,t] = x[:,t]
        if t>1:           
            Y_sf[:,:,t] = sigma**0.25*np.matmul(y[:,0:t],np.flip(phi[0:t,:], axis = 0))
            X_sf[:,:,t] = sigma**0.25*np.matmul(x[:,0:t],np.flip(phi[0:t,:], axis = 0))                               
            Features[n:n+filter_count*m,t] = np.reshape(Y_sf[:,:,t],(filter_count*m))
            Features[n+filter_count*m:,t] = np.reshape(X_sf[:,:,t],(filter_count*n))
         
        if t>1:
            m_sf[:,t] = regressor.predict(Features[:,t].reshape(1, -1)) 
        else:
            m_sf[:,t] = np.matmul(Theta[:,:,t-1],Features[:,t])
            
        # SLIP parameter update
        Z = np.matmul(Features[:,0:t+1],Features[:,0:t+1].T) # empirical covariance 
        z_invertible = matrix_norm(Z, -2) # returns smallest singular value of Z
        
        if np.min(Z) == 0 or t< 2:
            regularization = alpha_reg
        else:
            regularization = 0
        regressor = Ridge(alpha=regularization)
        regressor.fit(Features[:,0:t+1].T, y[:,0:t+1].T) # update the parameters by ridge regression
        Theta[:,:,t] = (regressor.coef_)
        
        # measure performance
        prediction_error[t,iteration] = np.linalg.norm(m_sf[:,t]-y[:,t])**2
        
        difference_error[t,iteration] = np.linalg.norm(m_sf[:,t]-m_kf[:,t])**2

        # if spectral radius of A > 1: measure *relative* error
        if A_spectral_radius > 1:
            prediction_error[t,iteration] /= np.linalg.norm(y[:,t])**2
            difference_error[t,iteration] /= np.linalg.norm(m_kf[:,t])**2
        if iteration == 0:
            iteration_duration = np.round(time.time() - t_start, 2)
            
        remaining_time = np.round(iteration_duration*(num_iterations - iteration - 1), 2)
        
    print('iteration = '+ str(iteration)+ ', '+ str(remaining_time)+ ' remaining until the end of simulation.')
#%% plot
if A_spectral_radius > 1:   
    y_label1 = '$/\|m_t\|^2$'
    y_label2 = '$/\|y_t\|^2$'
else:
    y_label1 = ' '
    y_label2 = ' '

if draw_plots:
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(np.mean(difference_error,axis = 1))
    plt.yscale('log')
    plt.ylabel (r'$||\hat{m}_t - m_t||^2$'+y_label1)
    plt.xlabel(r'$t$')
    plt.subplot(122)
    plt.plot(np.mean(prediction_error,axis = 1))
    plt.yscale('log')
    plt.ylabel (r'$||\hat{m}_t - y_t||^2$'+y_label2)
    plt.xlabel(r'$t$')
    plt.tight_layout()           