import numpy as np

def logistic_loss_and_grad(X, y, w):


    # Get logits
    N = X.shape[0]
    z = (X@w.reshape(-1,1)).reshape(-1)

    # Safely compute p
    p = np.zeros(N)
    p[z < 0] = np.exp(z[z<0])/(np.exp(z[z<0])+1)
    p[z >= 0] = 1/(1+np.exp(-z[z>=0]))

    # gradient
    grad = X.T @ (p-y) / N


    # Simplified loss
    ez = np.zeros(N)
    ez[z > 0] = np.exp(-z[z > 0])
    ez[z < 0] = 1/np.exp(z[z < 0])
    L = -(y*np.log(1)-z-np.log(1+ez)+z*y)

    return L, grad

    
if __name__ == '__main__':


    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    
    y = np.array([1.0, 0.0, 1.0])
    w = np.array([0.5, -0.25])

    print(logistic_loss_and_grad(X,y,w))