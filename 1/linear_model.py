import numpy as np
class LogisticRegression: 
    def __init__(self, lr=0.001, n_iters=1000):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = 1/(1+np.exp(-linear_pred))

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
        #assert X.shape[0] == y.shape[0]
        #assert len(X.shape) == 2
        # todo: implement

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = 1/(1+np.exp(-linear_pred))
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred