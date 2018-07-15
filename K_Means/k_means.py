import numpy as np

def dist(a, b):
    return np.sum(np.square(a - b), axis=-1)

class KMeans:
    def __init__(self, k):
        self.k = k

    def fit(self, data, eps=1e-4):
        """
        :param data: numpy array of shape (k, ..., dims)
        """
        self.dim = data.shape[-1]
        data = self.changeDims(data)
        self._initialize_means(data)
        
        
        labels, _ = self.predict(data)
        new_means = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
        
        while dist(self.means, new_means).sum() > eps:
            self.means = new_means
            labels, _ = self.predict(data)
            new_means = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])

    def _initialize_means(self, data):
        self.means = data[np.random.choice(data.shape[0], self.k)]
        
    def changeDims(self, data):
        return data.reshape([np.prod(data.shape[:-1]), self.dim])

    def predict(self, data):
        """
        :param data: numpy array of shape (k, ..., dims)
        :return: labels of each datapoint and it's mean
                 0 <= labels[i] <= k - 1
        """
        data = self.changeDims(data)
        distances = np.inf * np.ones(data.shape[0])
        labels = - np.ones(data.shape[0])
        
        for i,m in enumerate(self.means):
            cur_dists = dist(m, data)
            mask = cur_dists < distances
            labels[mask] = i
            distances[mask] = cur_dists[mask]
        
        return labels, self.means

class KMeansPlusPlus(KMeans):
    def _initialize_means(self, data):
        self.means = [data[np.random.choice(data.shape[0])]]
        for i in range(self.k-1):
            dists = np.ones(data.shape[0])
            for m in self.means:
                dists = np.minimum(dist(m, data), dists)
            self.means.append(data[np.random.choice(data.shape[0], p=dists/dists.sum())])
        self.means = np.asarray(self.means)

#         import matplotlib.pyplot as plt
#         plt.scatter(data[:,0], data[:, 1])
#         plt.scatter(self.means[:,0],self.means[:,1],c="r")
#         plt.show()