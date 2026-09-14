import numpy as np

def batched_logsumexp(A, axis=1):

    if axis == 0:
        A = A.T
        
    max_val = A.max(axis=1)
    
    A_shifted = A-max_val.reshape(-1,1)

    return max_val + np.log(np.sum(np.exp(A_shifted), axis=1))

if __name__ == '__main__':

    
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    A = np.array([
        [1000.0, 1001.0, 999.0],
        [-1000.0, -1001.0, -999.0]
    ])

    print(np.log(np.exp(A).sum(axis=0)))
    print(batched_logsumexp(A, axis=0))