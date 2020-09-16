import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

class Cov:
    def __init__(self, h=0.2):
        self.h = h 

    def gamma(self, s):
        return (np.power(np.abs(s - 1), 2 * self.h) + np.power(np.abs(s + 1), 2 * self.h) - 2 * np.power(np.abs(s), 2 * self.h))/2

class Structure:
    def __init__(self, sample_size=10):
        self.sample_size = sample_size
    
    def __str__(self):
        return f'the sample size {self.sample_size}, h paramater {self.h}'
    
    def matrix(self, cov_instance=Cov()):
        """return the covariance structure matrix here g(h, h) = 1"""
        n = self.sample_size
        cov = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                diff = i - j
                cov[i][j] = cov_instance.gamma(diff)
        return cov

class GaussianProcess:
    """
    attributes
    ----------
    cov_structure: stationnary covariance of multifractional gaussian process
    n: number of realizations of Y we want to generate
    cov_instance: pattern vector generating covariance matrix 
    methods
    -------
    get_params: get circular matrix and integer parameter m 
    spectrum: eigenvalues of circular matrix
    simulate_gaussian_process: main function
    slice_circular: generate circular matrix
    embedding_length: m parameter 
    check_symmetric: return True if arg is a symetric matrix
    """
    def __init__(self, cov_structure, cov_instance=Cov()):
        self.__cov_structure = cov_structure
        self.n = cov_structure.shape[0]
        self.cov_instance = cov_instance

    def __get_params(self, **kwargs):
        if kwargs:
            m = kwargs['embedding_value'] * 2
        else:
            m = self.__embedding_length(n=self.n)
        # get circular matrix pattern
        gamma0 = self.cov_structure[0,:] 
        theta0 = np.array([self.cov_instance.gamma(k) for k in range(m // 2 + 1)] + [self.cov_instance.gamma(k) for k in range(m // 2 + 1 , m)])
        return self.__slice_circular(theta0), m


    def __spectrum(self):
        """compute eigenvalues and eigenvectors of linear operator"""
        circular_matrix, m = self.__get_params()
        eigenvalues, eigenvectors = np.linalg.eig(circular_matrix)

        while not np.all(eigenvalues >= 0):
            circular_matrix, m = self.__get_params(embedding_value=m)
            eigenvalues, eigenvectors = np.linalg.eig(circular_matrix)
        return eigenvalues, eigenvectors
    
    def simulate_gaussian_process(self):
        """simulate the gaussian process"""
        eigenvalues, Q = self.__spectrum()
        gamma = np.diag(np.sqrt(eigenvalues))
        random_normal = np.random.normal(0, 1, gamma.shape[0]).reshape(-1, 1)
        a = gamma.dot(Q.T).dot(random_normal).flatten()
        return np.real(np.fft.fft(a)[:self.n]) 

    def generateFBM(self, delta, gp):
        coeff = np.power(delta, self.cov_instance.h)
        return np.array([coeff * gp[:i+1].sum() for i in range(self.n)])

    @staticmethod
    def __slice_circular(vector):
        """ generate circular matrix from a list pattern as first row"""
        ntimes = len(vector) - 1
        matrix = vector
        permute = vector.tolist()
        for i in range(ntimes):
            permute = [permute[-1]] + permute[:-1]
            matrix = np.vstack([matrix, permute])
        return matrix

    @staticmethod
    def __embedding_length(n):
      """for circular embedding matrix calculation
      return_back arg enable increse the power of g in 
      computed eigenvalues are negative
      """
      return np.power(2, int(1 + np.log(n - 1) / np.log(2)) + 1)

    @staticmethod 
    def __check_symmetric(a, tol=1e-8):
        """
        return if a given matrix is symetric or not with given confidence
        intervall
        """
        return np.all(np.abs(a-a.T) < tol)

    @property 
    def cov_structure(self):
      return self.__cov_structure 

    @cov_structure.setter
    def cov_structure(self, new_value):
        if isinstance(new_value, np.ndarray):
            row, col = new_value.shape
            is_symetric = self.__check_symmetric(new_value)
            is_square = row == col
            if is_square and is_symetric:
                self.__cov_structure = new_value
            else:
                raise('Give well define Matrix -- SPD')
        else:
            raise('Except ndarray object')


class MFBClass:
    def __init__(self, **kwargs):
        self.hurst_params = None 
        if kwargs:
            self.method = kwargs.get('method')
            self.size = kwargs.get('size')
            self.hurst_params = kwargs.get('params')
        else:
            raise('give params or func')

    @staticmethod
    def linear_hurst(t):
        return t
    
    @staticmethod
    def logistic_hurst(t):
        return 0.3 + 0.3 / (1 + np.exp(-100 * (t - 0.7)))

    @staticmethod
    def periodic_hurst(t):
        return 0.5 + 0.49 * np.sin(4 * np.pi * t)

    def simulate(self):
        if self.hurst_params is None:
            if not isinstance(self.size, int):
                raise('size parameter must be int')

            grid = np.linspace(0.001, 0.99, self.size)
            if self.method == 'linear':
                hurst_func = np.vectorize(self.linear_hurst)
            elif self.method == 'logistic':
                hurst_func = np.vectorize(self.logistic_hurst)
            elif self.method == 'periodic':
                hurst_func = np.vectorize(self.periodic_hurst)
            else:
                raise('parameter value given does not fit with any default value')
            self.hurst_params = hurst_func(grid)

        mfb_vect = []
        for hurst_param in self.hurst_params:
            cis = Cov(h=hurst_param)
            matrix = Structure().matrix(cis)
            simu = GaussianProcess(matrix, cis)
            gp = simu.simulate_gaussian_process()
            brm = simu.generateFBM(delta=10, gp=gp)
            mfb_vect.append(brm)
        return mfb_vect
            