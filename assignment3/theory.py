import numpy as np
import scipy.ndimage as scip

A = np.array([[2,1,2,3,1],
             [3,9,1,1,4],
             [4,5,0,7,0]])

B = np.array([[-1,0,1],
             [-2,0,2],
             [-1,0,1]])

res = scip.convolve(A, B, mode='constant')

print(res)