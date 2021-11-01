import numpy as np
import sys

#-------------------------------------------------------------------#
# Solve Schrodinger equation using the numerical discretisation     #
# method of Chung-Phillips, J. Comput. Chem., 1992, 13, 874-882     #
# Inputs:  Potential and moments of inertia on finely-spaced grids  # 
# Outputs: Vibrational state energies in atomic units               #
#-------------------------------------------------------------------#

def cp(V,I):

  N = len(V)
  h = 2*np.pi/(N-1)
  H = np.zeros((N,N))
 
  # Fill in diagonal elements of discretized Hamiltonian matrix
  for i in range(0,N):
    H[i,i] = -2.0/I[i] -2.0*h*h*V[i] 

  # Fill in off-diagonal elements
  for i in range(1,N-1):
    H[i,i-1] = 2.0/(I[i]+I[i-1])
    H[i,i+1] = 2.0/(I[i]+I[i+1])
    H[i-1,i] = H[i,i-1]
    H[i+1,i] = H[i,i+1]
  H[1,N-1] = H[1,0]
  H[N-1,N-2] = H[N-2,N-1]
  H[N-1,1] = H[1,N-1]

  # Diagonalise H, dropping first row and column (same as last) and reinstating later
  eigvals,eigvecs = np.linalg.eigh(H[1:,1:])
  unique_eigvals, unique_indices = np.unique(np.round(eigvals/(-2*h*h),8),return_index=True)
  last_row = eigvecs[-1,:]
  wfn = np.insert(eigvecs,0,last_row,axis=0)
  wfn_norm = np.sum(np.square(wfn),axis=0)
  wfn_norm_array = np.outer(np.ones(N),wfn_norm)
  normalised_wfn = wfn / wfn_norm_array
  reverse = [N-1-i for i in range(1,N)]

  return eigvals, unique_eigvals, normalised_wfn[:,reverse]

