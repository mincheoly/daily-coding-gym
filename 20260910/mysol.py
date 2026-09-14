import numpy as np


def pairwise_sq_distances(X, Y):

    X_sq = (X**2).sum(axis=1).reshape(-1,1)
    Y_sq = (Y**2).sum(axis=1).reshape(1,-1)
    XY = X@Y.T

    D2 = X_sq - 2*XY + Y_sq

    D2[D2 < 0] = 0

    return D2


def nearest_neighbor_distances(X, Y, chunk_size=1024):

    chunk_idx = 0
    num_chunks = int(np.ceil(X.shape[0]/chunk_size))

    distances = np.zeros(X.shape[0])

    for chunk_idx in range(num_chunks):

        rowstart, rowend = (chunk_size*(chunk_idx)),(chunk_size*(chunk_idx+1))

        X_chunk = X[rowstart:rowend]

        distances[rowstart:rowend] = np.sqrt(pairwise_sq_distances(X_chunk,Y).min(axis=1))

    return distances

if __name__ == '__main__':

    N = 10
    D = 10
    M = 7
    X = np.random.random((N, D))
    Y = np.random.random((M, D))
    
    distances = np.sqrt(((X[:, None, :] - Y[None, :, :])**2).sum(axis=2))
    result = distances.min(axis=1)

    print( result )
    print( nearest_neighbor_distances(X, Y, chunk_size=1024) )