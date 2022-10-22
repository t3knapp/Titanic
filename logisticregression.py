import numpy as np

class Logistic(object):
    """Titanic V1 - Binary Classifier using Logistic Regression

    Parameters
    -----------
    eta : float
        Learning rate between 0.0 and 1.0
    n_iter : int
        passes over the training dataset

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications after each epoch
    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit the training data with logistic regression

        Parameters
        ----------
        X : {array-like} = n_samples by n_features
        Training vectors
        y: {array-like} = n_samples
        Target values

        Returns
        ---------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.sigmoid(self.net_input(X))
            errors_ = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors_)
            self.w_[0] += self.eta * errors_.sum()

    def sigmoid(self, z):
        """Sigmoid function used in logistic regression
        Parameters
        ----------
        z: {array-like} = n_samples by n_features
        Training vectors for single feature

        """
        return np.divide(np.ones(z.shape), np.ones(z.shape) + np.exp(z))

    def net_input(self, X):
        """Calculating net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after the unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)