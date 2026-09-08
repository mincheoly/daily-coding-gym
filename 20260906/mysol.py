import numpy as np

def pairwise_sq_distances(X, Y):

    X_sq = (X**2).sum(axis=1).reshape(-1,1)
    Y_sq = (Y**2).sum(axis=1).reshape(1,-1)
    XY = X@Y.T

    D2 = X_sq - 2*XY + Y_sq

    D2[D2 < 0] = 0

    return D2


if __name__ == '__main__':

    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    
    Y = np.array([
        [0.0, 1.0],
        [2.0, 0.0],
    ])

    # X = np.array([
    #     [10000.0, 0.001],
    #     [-10000.0, 0.002],
    # ])
    
    # Y = np.array([
    #     [9999.0, 0.003],
    #     [0.0, 0.0],
    # ])

    print(pairwise_sq_distances(X,Y))