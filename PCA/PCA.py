import numpy as np

class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, data):
        """
        finds best params for X = Mu + A * Lambda
        :param data: data of shape (number of samples, number of features)
        HINT! use SVD
        """
        Data = np.asarray(data)
        self.location_ = Data.mean(axis=0)
        Data -= self.location_ 
        u,s,v = np.linalg.svd(Data)
        indexes = s.argsort()[-self.k:][::-1]
        self.basis = v[indexes]
        
    def getBasis(self):
        return self.basis

    def transform(self, data):
        """
        for given data returns Lambdas
        x_i = mu + A dot lambda_i
        where mu is location_, A is matrix_ and lambdas are projection of x_i
        on linear space from A's rows as basis
        :param data: data of shape (number of samples, number of features)
        """
        Data = np.asarray(data - self.location_)
        return self.basis * Data.T