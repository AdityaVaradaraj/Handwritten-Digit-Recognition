import numpy as np

def logistic_regression(X,y, lr, n_epochs):
    N,D = X.shape
    N,M = y.shape
    Theta = np.zeros((D,M))
    Phi = np.zeros((N,M))
    print('N, D, M: ', N, D, M)
    losses = []
    for i in range(n_epochs):
        # Calculate Posteriors
        Phi = (np.exp(X.dot(Theta)) / np.sum(np.exp(X.dot(Theta)), axis=1).reshape(N,1))
        # Corr. Loss
        loss = -1*np.sum(y * np.log(Phi))
        # Append loss
        losses.append(loss)
        # Calculate Gradients 
        grads = np.matmul(X.T, Phi - y)
        # Gradient Descent Update
        Theta = Theta - lr*grads
        
    # Calculate Posteriors
    Phi = (np.exp(X.dot(Theta)) / np.sum(np.exp(X.dot(Theta)), axis=1).reshape(N,1))
    # Corr. Loss
    loss = -1*np.sum(y * np.log(Phi))
    # Append loss
    losses.append(loss)
    y_pred = np.argmax(Phi, axis = 1)
    return y_pred, losses, Theta