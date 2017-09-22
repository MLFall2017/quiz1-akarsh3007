
import numpy as np

def PCA(input) :

    cov_in = np.corrcoef(input.T)
    eig_val, eig_vec = np.linalg.eig(cov_in)
    eig_pairs = [(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort()
    eig_pairs.reverse()
    tot = sum(eig_val)
    eigval_per = [(i/tot)*100 for i in sorted(eig_val,reverse=True)]
    cum_sum = np.cumsum(eigval_per)
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),eig_pairs[1][1].reshape(3,1)))
    reduced = input.dot(matrix_w)
    return reduced
