
# EDGE Estimator for Shannon Mutual Information
#
# Created by Morteza Noshad (noshad@umich.edu)
#
# Based on the paper: Scalable Mutual Information Estimation using Dependence Graphs
#
################
# The estimator is in the following form:
#
# I = EDGE(X,Y,U=u,gamma=[gamma_X,gamma_Y], epsilon=[eps_X, eps_Y], hashing)
#
# Arguments: 
#
# X is N * d_x and Y is N * d_Y data sets
# U (optional) is an upper bound on the MI. It doesn't need to be accurate, but more accurate upper bound we set, faster convergence rates we get
# gamma=[gamma_X,gamma_Y] (optional) is the vector of soothness for X and Y. 
#		For example, if the data is discrete we set gamma close to 0, 
#		and if the data is continuous we set gamma close to 1 (or maybe higher if it is very smooth) 
# epsilon=[eps_X, eps_Y] (optional) is the vector of bandwidths for X and Y. If no epsilon is set, 
#		automatic bandwidths will be set.
# hashing (optional): possible arguments are 'p-stable' (default) which is a common type of LSH
#		or 'floor' which uses the simple floor function as hashing
# 
# Output: I is the estimation of mutual information between X snd Y 
###########################

import numpy as np
import math
#import cvxpy as cvx # Need to install CVXPY package, 
					#  it is also possible to run this code without cvxpy, by setting uniform weights 
import time
from scipy.special import *
from sklearn.neighbors import NearestNeighbors
#from random import randint, seed
#np.random.seed(seed=0)

#####################
#####################

# Generate W and V matrices (used in LSH)
def gen_W(X,Y):
	np.random.seed(3334)
	# Num of Samples and dimensions
	N = X.shape[0]
	dim_X , dim_Y  = X.shape[1], Y.shape[1]

	# parameters to control the dimension of W and V 
	kx,ky = 2, 2
	rx,ry = 10,10

	# Find standard deviation vectors
	# print()
	# print("X shape:", X.shape, "dim_X: ", dim_X)
	# print("Y shape:", Y.shape, "dim_Y: ", dim_Y)
	# print()
	std_X = np.array([np.std(X[:,[i]]) for i in range(dim_X)]) 
	std_Y = np.array([np.std(Y[:,[i]]) for i in range(dim_Y)])

	std_X = np.reshape(std_X,(dim_X,1))
	std_Y = np.reshape(std_Y,(dim_Y,1))

	# Compute dimensions of W and V
	d_X_shrink=min(dim_X,math.floor(math.log(1.0*N/rx,kx)))
	d_Y_shrink=min(dim_Y,math.floor(math.log(1.0*N/ry,ky)))

	# # HERE
	# print("dim X: ", dim_X)
	# print("dim_Y: ", dim_Y)
	# print("d_X_shrink: ", d_X_shrink)
	# print("d_Y_shrink: ", d_Y_shrink)
	
	# Repeat columns of std_X and Y to be in the same size as W and V
	std_X_mat= np.tile(std_X,(1,d_X_shrink))
	std_Y_mat= np.tile(std_Y,(1,d_Y_shrink))

	# avoid devision by zero
	std_X_mat[std_X_mat<0.0001]=1
	std_Y_mat[std_Y_mat<0.0001]=1

	# Mean and standard deviation of Normal pdf for elements of W and V
	mu_X = np.zeros((dim_X, d_X_shrink)) 
	mu_Y = np.zeros((dim_Y, d_Y_shrink)) 

	sigma_X = 1.0/(std_X_mat *np.sqrt(dim_X))
	sigma_Y = 1.0/(std_Y_mat *np.sqrt(dim_Y)) 

	# Generate normal matrices W and V
	#np.random.seed(seed=0)
	W = np.random.normal(mu_X, sigma_X, (dim_X, d_X_shrink))
	V = np.random.normal(mu_Y, sigma_Y, (dim_Y, d_Y_shrink))

	return (W,V)

# Find KNN distances for a number of samples for normalizing bandwidth
def find_knn(A,d):
	np.random.seed(3334)
	#np.random.seed()
	#np.random.seed(seed=int(time.time()))
	r = 500
	# random samples from A
	A = A.reshape((-1,1))
	N = A.shape[0]
	
	k=math.floor(0.43*N**(2/3 + 0.17*(d/(d+1)) )*math.exp(-1.0/np.max([10000, d**4])))

	T= np.random.choice(A.reshape(-1,), size=r).reshape(-1,1)
	nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A)
	distances, indices = nbrs.kneighbors(T)
	d = np.mean(distances[:,-1])
	return d

# Returns epsilon and random shifts b
def gen_eps(XW,YV):
	d_X , d_Y  = XW.shape[1], YV.shape[1]
	# Find KNN distances for a number of samples for normalizing bandwidth
	eps_X = np.array([find_knn(XW[:,[i]],d_X) for i in range(d_X)]) + 0.0001
	eps_Y = np.array([find_knn(YV[:,[i]],d_Y) for i in range(d_Y)]) + 0.0001

	return (eps_X,eps_Y)

# Define H1 (LSH) for a vector X (X is just one sample)
def H1(XW,b,eps):
	
	# dimension of X
	d_X = XW.shape[0]
	#d_W = W.shape[1]
	XW=XW.reshape(1,d_X)

	# If not scalar
	if d_X > 1:
		X_te = 1.0*(np.squeeze(XW)+b)/eps	
	elif eps>0:
		X_te = 1.0*(XW+b)/eps
	else:
		X_te=XW

	# Discretize X
	X_t = np.floor(X_te)
	if d_X>1: 
		R = tuple(X_t.tolist())
	else: R = np.squeeze(X_t).item() # HERE R=np.asscalar(np.squeeze(X_t))
	return R

# Compuate Hashing: Compute the number of collisions in each bucket
def Hash(XW,YV,eps_X,eps_Y,b_X,b_Y):

	# Num of Samples and dimensions
	N = XW.shape[0]

	# Hash vectors as dictionaries
	CX, CY, CXY = {}, {}, {} 
	
	# Computing Collisions
	
	for i in range(N):
		# Compute H_1 hashing of X_i and Y_i: Convert to tuple (vectors cannot be taken as keys in dict)

		X_l, Y_l = H1(XW[i],b_X,eps_X), H1(YV[i],b_Y,eps_Y)

		# X collisions: compute H_2 
		if X_l in CX:
			CX[X_l].append(i)
		else: 
			CX[X_l] = [i]
			
		# Y collisions: compute H_2
		if Y_l in CY:
			CY[Y_l].append(i)
		else: 
			CY[Y_l] = [i]

		# XY collisions
		if (X_l,Y_l) in CXY:
			CXY[(X_l,Y_l)].append(i)
		else: 
			CXY[(X_l,Y_l)] = [i]

	return (CX, CY, CXY)


# Compute mutual information and gradient given epsilons and radom shifts
def Compute_MI(XW,YV,U,eps_X,eps_Y,b_X,b_Y):
	N = XW.shape[0]

	(CX, CY, CXY) = Hash(XW,YV,eps_X,eps_Y,b_X,b_Y)

	# Computing Mutual Information Function
	I = 0
	N_c = 0
	for e in CXY.keys():
		Ni, Mj, Nij = len(CX[e[0]]), len(CY[e[1]]), len(CXY[e])

		if 1==1: 
			I += Nij* max(min(math.log(1.0*Nij*N/(Ni*Mj),2), U),0.001)
			N_c+=Nij

	I = 1.0* I / N_c

	return I

def EDGE(X,Y,U=10, gamma=[1, 1], epsilon=[0,0], epsilon_vector = 'fixed', eps_range_factor=0.1, normalize_epsilon = True ,
				ensemble_estimation = 'median', L_ensemble=5 ,hashing='p-stable', stochastic = False):
	
	gamma = np.array(gamma)
	epsilon = np.array(epsilon)

	if X.ndim==1:
		X=X.reshape((-1,1))
	if Y.ndim==1:
		Y=Y.reshape((-1,1))
	# Num of Samples and dim
	N, d = X.shape[0], X.shape[1]

	# Find dimensions
	dim_X, dim_Y  = X.shape[1], Y.shape[1]
	dim = dim_X + dim_Y


## Hash type

	if hashing == 'p-stable':	
		# Generate random transformation matrices W and V 
		# # HERE
		# print("X shape:", X.shape)
		# print("Y shape:", Y.shape)
		(W,V) = gen_W(X,Y)
		d_X_shrink, d_Y_shrink=W.shape[1], V.shape[1]
		# Find inner products
		XW, YV = np.dot(X,W), np.dot(Y,V)
	
	elif hashing == 'floor':
		#W = np.identity(dim_X)	
		#V = np.identity(dim_Y)
		d_X_shrink, d_Y_shrink=dim_X, dim_Y 
		XW, YV = X, Y
	
## Initial epsilon and apply smoothness gamma

	# If no manual epsilon is set for computing MI:
	if epsilon[0] ==0:
		# Generate auto epsilon and b
		(eps_X_temp,eps_Y_temp) = gen_eps(XW,YV)

		# Normalizing factors for the bandwidths
		cx, cy = 3*d_X_shrink, 3*d_Y_shrink
		eps_X0, eps_Y0 = eps_X_temp * cx*gamma[0], eps_Y_temp * cy*gamma[1] 
	else:
		eps_X_temp = np.ones(d_X_shrink,)*epsilon[0]
		eps_Y_temp = np.ones(d_Y_shrink,)*epsilon[1]	
		cx, cy = 3*d_X_shrink, 3*d_Y_shrink
		eps_X0, eps_Y0 = eps_X_temp * cx*gamma[0], eps_Y_temp * cy*gamma[1] 

## epsilon_vector
	if epsilon_vector == 'fixed':
		T = np.ones(L_ensemble)
	elif epsilon_vector == 'range':
		T = np.linspace(1,1+eps_range_factor,L_ensemble)


## Compute MI Vector
	
	# MI Vector
	I_vec = np.zeros(L_ensemble)

	for j in range(L_ensemble):

		# Apply epsilon_vector 
		eps_X, eps_Y = eps_X0 * T[j], eps_Y0 * T[j]
			
		## Shifts of hashing
		if stochastic== True:
			np.random.seed()
			f=0.1
			b_X = f*np.random.rand(d_X_shrink,)*eps_X
			b_Y = f*np.random.rand(d_Y_shrink,)*eps_Y
		else:
			b_X = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_X
			b_Y = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_Y

		I_vec[j] = Compute_MI(XW,YV,U,eps_X,eps_Y,b_X,b_Y)

## Ensemble method

	if ensemble_estimation == 'average':
		I = np.mean(I_vec) 
	# elif ensemble_estimation == 'optimal_weights': 
	# 	weights = compute_weights(L_ensemble, d, T, N)
	# 	weights=weights.reshape(L_ensemble,)
	# 	I = np.dot(I_vec, weights)
	elif ensemble_estimation == 'median':
		I = np.median(I_vec)
	 	

## Normalize epsilon according to MI estimation (cross validation)
	if normalize_epsilon == True:
		gamma=gamma * math.pow(2,-math.sqrt(I*2.0)+(0.5/I))
		normalize_epsilon = False
		I = EDGE(X,Y,U, gamma, epsilon, epsilon_vector, eps_range_factor, normalize_epsilon, ensemble_estimation, L_ensemble,hashing, stochastic)


	return I

####################################
####################################

# if __name__ == "__main__":

# # Fully Dependent Datasets
# 	X = np.random.rand(10000,50)
# 	Y = X+0.001*np.random.rand(10000,50)

# 	#I = EDGE(X,Y,U=10,gamma=[1, 1],epsilon=(0.2,0.2)) # Estimated Mutual Information between X and Y using EDGE method
# 	#I = EDGE(X,Y,U=100,epsilon=[1,1])
# 	I = EDGE(X,Y)

# 	print ('Estimated MI ', I)
# 	print('################################')
	