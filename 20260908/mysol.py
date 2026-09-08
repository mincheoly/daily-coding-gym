import numpy as np

def streaming_mean_var(chunks):

    s = 0
    s_sq = 0
    N = 0
    
    for X in chunks:

        s += X.sum(axis=0)
        s_sq += (X**2).sum(axis=0)
        N += X.shape[0]
        
    m = s/N

    return m, s_sq/(N-1) - (m**2)*N/(N-1)
        
if __name__ == '__main__':

    D = 2
    chunks = [np.random.random((np.random.randint(1,5), D)) for i in range(5)]

    # Naive
    X = np.concatenate(chunks, axis=0)
    mean = X.mean(axis=0)
    var = X.var(axis=0, ddof=1)
    print('naive', mean, var)

    # My implementation

    print('mine', *streaming_mean_var(chunks))