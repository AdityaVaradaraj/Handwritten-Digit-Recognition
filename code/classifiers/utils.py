import numpy as np


def flatten(X):
    # X : N x H x W
    X = X.reshape(X.shape[0], -1)
    return X

def y_encoding(y,M):
    N = y.shape[0]
    y_matrix = np.zeros((N,M))
    for n in range(N):
        y_matrix[n, y[n]] = 1
    return y_matrix

def PCA_proj(X):
    print('---------------- PCA --------------------')
    # X: N x D
    N, D = X.shape
    print(N,D)
    original_mean_img = np.mean(X, axis=0)
    # ------------ Center the data -----------------
    print('Data Centering.....')
    X = X - np.mean(X, axis=0)
    print('Data Centered')
    # ------------ Find Covariance Matrix -----------
    Sigma = 0
    Sigma = (N-1)/ N * np.cov(X, rowvar = False)
    print('Covariance calculated')
    # ------------ Find Eigenvalues and eigenvectors ------------
    W, V = np.linalg.eig(Sigma)
    print('Eigvals and Eigvecs calculated')
    # ------------ Find Required no.of components --------------- 
    keep_var = 0.99 # Minimum fraction of Variance to keep
    req_var = keep_var*sum(W)
    req_dim = 0
    variance = 0
    for i in range(len(W)):
      variance += np.abs(W[i])
      if variance >= req_var:
          req_dim = i + 1
          m = req_dim
          break
    print('Required Dimension: ', m)
    # ------------ Sort eigenvectors in decreasing order and take top "m" ------------
    idx = np.argsort(np.real(W))[::-1]
    V = V[:,idx]
    V_new = V[:, :m]
    print('Got new clipped projection matrix')
    # Take Projection in the m eigenvector directions
    X_new = np.real(np.matmul(X, V_new))
    print('X_train projected')
    return X_new, V_new

def MDA(X, y, M):
  D = X.shape[1] # No. of features
  N = y.shape[0] # No. of samples
  means = []
  priors = np.zeros((M,1))
  anchor_mean = np.zeros((D,1))
  Sigma = []
  
  # ------------ Estimate class mean vectors, covariance matrices and priors -------
  for i in range(M):
      Ni = np.count_nonzero(y==i)
      class_ind = np.where(y==i)[0]
      
      priors[i] = Ni/N
      mean_i = (1/Ni)*X[class_ind, :].sum(axis=0).reshape(D,1)
      means.append(mean_i)
      anchor_mean += priors[i]*means[i]
      Sigma_i = (Ni-1)/Ni * np.cov(X[class_ind, :], rowvar = False) 
      Sigma.append(Sigma_i)  
  # ------------ Find Between class and Within class Scatter ------------
  Sigma_b = np.zeros((D,D))
  Sigma_w = np.zeros((D,D))
  for i in range(M):
      Sigma_b += priors[i] * np.matmul(means[i] - anchor_mean, (means[i] - anchor_mean).T)
      Sigma_w += priors[i] * Sigma[i]
  if np.linalg.det(Sigma_w) == 0:
      Sigma_w += 0.0001*np.eye(D)
  
  # ------------- Top m Eigenvectors of Sigma_w^(-1) Sigma_b ------------
  W, V = np.linalg.eig(np.matmul(np.linalg.inv(Sigma_w), Sigma_b))
  m = np.count_nonzero(np.real(W) > 1e-10) # m <= M-1, where, M is no. of classes
  idx = np.argsort(np.real(W))[::-1]
  sorted_V = V[:,idx]
  A = sorted_V[:,:m]
  # Find Theta by dividing by no. of features
  Theta = (1/D)*A
  print('Theta shape: ', Theta.shape)
  # ------------- Project the Data ----------------
  print('------------ MDA -------------')
  Z = np.matmul(X,  Theta)
  
  return np.real(Z), Theta


