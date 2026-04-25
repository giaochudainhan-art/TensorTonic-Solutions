import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    (N,D)= X.shape 
    W = np.zeros(D)
    B= 0.0 
    for t in range(steps):
        z= np.dot(W,X.T)+B
        y_pred = 1/(1+np.exp(-z))
        # Caculte the gradient
        errors = y_pred - y 
        dW = np.dot(X.T, errors)/N
        dB = np.sum(errors)/N
        # Update weight 
        W -= lr*dW
        B -= lr*dB
    return (W, float(B))