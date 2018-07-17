import scipy.stats as st
import sys
sys.path.append("..")
from K_Means.k_means import KMeansPlusPlus

class GMM:
    def __init__(self, k):
        self.k = k
        self.means = []
        self.covariances = []
        self.pis = []
        self.gammas = []

    def fit(self, data, eps=1e-6):
        """
        :params data: np.array of shape (..., dim)
                                  where dim is number of dimensions of point
        """
        data = np.array(data, np.float)
        self._initialize_params(data)
        
        old_means = self.get_means()
        self._E_step(data)
        self._M_step(data)
        
        while np.sum(np.square(old_means - self.means)) > eps:
            old_means = self.get_means()
            self._E_step(data)
            self._M_step(data)
            
    def _initialize_params(self, data):
        
        kmpp = KMeansPlusPlus(self.k)
        kmpp.fit(data)
        
        _, self.means = kmpp.predict(data[:1])
        
        self.covariances = np.array([np.identity(data.shape[-1])] * self.k) 
        self.pis = np.ones(self.k, np.float)/self.k
        

    def _E_step(self, data):
        normalpdfs = [st.multivariate_normal(self.means[i], self.covariances[i]).pdf
                      for i in range(self.k)]
        self.gammas = np.zeros((data.shape[0], self.k), np.float)
        for i in range(data.shape[0]):
            
            self.gammas[i] = self.pis * np.array([pdf(data[i]) for pdf in normalpdfs])
            self.gammas[i] /= self.gammas[i].sum()

    def _M_step(self, data):
        gammaSums = self.gammas.sum(axis=0)
        self.means = self.gammas.T.dot(data) /gammaSums[None].T
        
        self.covariances = np.zeros_like(self.covariances)
        for j in range(self.k):
            for i in range(data.shape[0]):
                a = (data[i] - self.means[j])
                self.covariances[j] += self.gammas[i,j] * np.outer(a, a)
            self.covariances[j] /= gammaSums[j]
            
        self.pis = gammaSums/gammaSums.sum()
        
    def predict(self, data):
        """
        :param data: np.array of shape (..., dim)
        :return: np.array of shape (...) without dims
                         each element is integer from 0 to k-1
        """
        normalpdfs = [st.multivariate_normal(self.means[i], self.covariances[i]).pdf
                      for i in range(self.k)]
        return np.array([self._pred(d, normalpdfs) for d in data])

    def _pred(self, x, normalpdfs):
        return np.argmax([pdf(x) for pdf in normalpdfs])
        
        
    def get_means(self):
        return self.means.copy()

    def get_covariances(self):
        return self.covariances.copy()

    def get_pis(self):
        return self.pis.copy()