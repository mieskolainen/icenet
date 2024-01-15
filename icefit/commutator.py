# Toy simulation between selection orders A and B aka "commutation test",
# when two variables (X,Y) are correlated.
#
# Run with: python commutator.py
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import scipy
                
def selection_A(X, Y, Y_CUT):
    "Per event: Cut on Y and then get the min(X) object"
    "Returns: Selected object Y value"
    passing = X * (Y > Y_CUT)
    if np.count_nonzero(passing) != 0:
        return Y[np.argmin(passing)]
    else:
        return False

def selection_B(X, Y, Y_CUT):
    "Per event: Get the min(X) object and then cut on Y"
    "Returns: Selected object Y value"
    k = np.argmin(X)
    if Y[k] > Y_CUT:
        return Y[k]
    else:
        return False

def crv(size, upper_L):
    return np.random.normal(0.0, 1.0, size=size) @ upper_L

def main():
    
    Y_CUT  = 0.0      # 0 puts the cut at the median because of 0-centered Gaussian RVs
    N      = int(1e5) # Number of events
    maxObj = 6        # Maximum number of objects
    rhoval = [-0.99, -0.5, 0.0, 0.5, 0.99]

    # Level of correlation
    for rho in rhoval:
        
        cor_matrix = np.array([[1.0, rho],
                               [rho, 1.0]])
        
        upper_L = np.transpose(np.linalg.cholesky(cor_matrix))
        
        # Check the Cholesky matrix is right way around
        rv      = crv(size=(int(1e5), 2), upper_L=upper_L)
        print(f'Correlation rho(X,Y): {scipy.stats.pearsonr(rv[:,0], rv[:,1])[0]:0.2f}')
        
        # Different number of objects
        for nObj in range(1,maxObj+1):
            
            N_A, N_B, N_AB = 0,0,0
            
            # Event loop
            for i in range(N):
                
                # Build correlated Gaussian random variables
                rv   = crv(size=(nObj, 2), upper_L=upper_L)
                X, Y = rv[:,0], rv[:,1]
                
                a = selection_A(X=X, Y=Y, Y_CUT=Y_CUT)
                b = selection_B(X=X, Y=Y, Y_CUT=Y_CUT)
                
                if a != False: N_A  += 1
                if b != False: N_B  += 1
                if a == b:     N_AB += 1
            
            print(f'[nObj: {nObj:2d}] sel(A) event rate: {N_A/N:0.3f}, sel(B) event rate: {N_B/N:0.3f}, sel(A) == sel(B) rate: {N_AB/N:0.3f}')
        print('')


if __name__ == "__main__":
    main()
