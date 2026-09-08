import numpy as np

def softmax_rows(X):

    lse = np.log(np.exp(X).sum(axis=1)).reshape(-1,1)
    print(lse)
    logp = X - lse
    return np.exp(logp)

if __name__ == '__main__':

    X = np.array([
    [1000.0, 1001.0, 1002.0],
    [-1000.0, -1001.0, -1002.0],
    ])
    print(softmax_rows(X))


# Solution:
# def softmax_rows(X):
#     X_shifted = X - X.max(axis=1, keepdims=True)
#     exp_X = np.exp(X_shifted)
#     return exp_X / exp_X.sum(axis=1, keepdims=True)
    
# Take home point: 
# Sutract a constant (like max of the array) prior to exponentiation when it's going to be normalized out anyways.