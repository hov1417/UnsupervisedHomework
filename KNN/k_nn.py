import numpy as np
from collections import Counter

def dist(a, b)
    return np.sum(np.square(a-b))

class K_NN:
    def __init__(self, k):
        """
        :param k: number of nearest neighbours
        """
        self.k = k

    def fit(self, data):
        """
        :param data: 3D array, where data[i, j] is i-th classes j-th point (vector: D dimenstions)
        """
        self.data = np.asarray(data)

    def predict(self, X):
        """
        :param data: 2D array of floats N points each D dimensions
        :return: array of integers
        """
        res=[]
        for x in X:
            dists = []
            for i in range(self.data.shape[0]):
                for d in self.data[i]:
                    dists.append((dist(d, x), i))
            nearests = [a[1] for a in sorted(dists)[:self.k]]
            
            counts = dict(Counter(nearests))
            print(counts)
#             res.append(
        return 0
