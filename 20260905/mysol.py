import numpy as np

def streaming_logsumexp(chunks):

    sum_so_far = 0

    a = 0
    A = 0
    
    for x in chunks:
        # chunk is a 1d np array
        b = x.max()
        B = np.sum(np.exp(x-b))

        c = max(a,b)
        C = np.exp(a-c)*A + np.exp(b-c)*B

        # update a and A with newly calculated values
        a = c
        A = C

    return a + np.log(A)


if __name__ == '__main__':

    chunks = iter([
        np.array([1.0, 2.0]),
        np.array([3.0]),
        np.array([4.0, 5.0]),
    ])

    print(streaming_logsumexp(chunks))