import numpy as np

class GaussianProcess:
    """
    attributes
    ----------
    cov_structure: stationnary covariance of multifractional gaussian process
    n: number of realizations of Y we want to generate

    methods
    -------
    get_params: 
    spectrum: 
    simulate_gaussian_process:
    slice_circular:
    embedding_length:
    check_symmetric:
    """
    def __init__(self, cov_structure, process_length=3):

        self.cov_structure = cov_structure
        self.n = process_length

    def __get_params(self, **kwargs):
        m = self.__embedding_length(n=self.n)
        if kwargs:
            m = kwargs['embedding_value'] * 2
        # get circular matrix pattern
        gamma0 = self.cov_structure[0,:] 
        theta0 = np.array([gamma0[k] for k in range(m//2 + 1)] + [gamma0[m-k] for k in range(m// 2 + 1, m)])
        return self.__slice_circular(theta0), m

    def __spectrum(self):
        """compute eigenvalues and eigenvectors of linear operator"""
        circular_matrix, m = self.__get_params()
        eigenvalues, eigenvectors = np.linalg.eig(circular_matrix)
        while not np.all(eigenvalues > 0):
            circular_matrix, m = self.__get_params(embedding_value=m)
            eigenvalues, eigenvectors = np.linalg.eig(circular_matrix)
        return eigenvalues, eigenvectors
    
    def simulate_gaussian_process(self):
        """simulate the gaussian process"""
        eigenvalues, Q = self.__spectrum()
        gamma = np.diag(np.sqrt(eigenvalues))
        random_normal = np.random.normal(0, 1, gamma.shape[0]).reshape(-1, 1)
        a = gamma.dot(Q.T).dot(random_normal).flatten()
        return np.fft.fft(a)[:self.n]

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
      computed eigenvalues are negative"""
      return int(1 + np.log(n - 1) / np.log(2))

    @staticmethod 
    def __check_symmetric(a, tol=1e-8):
        """return if a given matrix is symetric or not with given confidence
        intervall"""
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
                raise('Try to give well define Matrix -- SPD')

        else:
            raise('Except ndarray object')
    


