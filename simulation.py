class GaussianProcess:
  def __init__(self, cov_structure, process_length=7):
    self.cov_structure = cov_structure
    self.n = process_length

  @staticmethod
  def _embedding_length(n, return_back=False):
    """for circular embedding matrix calculation
    return_back arg enable increse the power of g in 
    computed eigenvalues are negative"""
    return int(1 + np.log(n - 1) / np.log(2)) + 1

  @staticmethod 
  def _check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


  @property 
  def cov_structure(self):
    return self.__cov_structure 

  @cov_structure.setter
  def cov_structure(self, new_value):
    if isinstance(new_value, np.ndarray):
      row, col = new_value.shape
      is_symetric = self._check_symmetric(new_value)
      is_square = row == col
      if is_square and is_symetric:
        self.__cov_structure = new_value
      else:
        raise('Try to give well define Matrix -- SPD')



