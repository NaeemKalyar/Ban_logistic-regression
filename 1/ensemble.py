from data_handler import bagging_sampler,load_dataset,split_dataset
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator,X_train,X_test,y_test,y_train):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.X_train=X_train
        self.X_test=X_test
        self.y_test=y_test
        self.y_train=y_train
        self.base_estimator=base_estimator
        self.n_estimator=n_estimator
        self.weights = None
        self.bias = None


    def fit(self, X, y,lr=0.001, n_iters=1000):
        """
        :param X:
        :param y:
        :return: self
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = 1/(1+np.exp(-linear_pred))

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - lr*dw
            self.bias = self.bias - lr*db
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        
        
        # todo: implement

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        list_pred=[]
        for i in range(self.n_estimator):
            X_trains, y_trains=bagging_sampler(self.X_train,self.y_train)
            classifier=self.base_estimator
            classifier.fit(X_trains,y_trains)
            pred=classifier.predict(self.X_test)
            list_pred.append(pred)
        pred=[]
        for i in range(len(self.y_test)):
            
            sum=0
            for j in range(self.n_estimator):
                sum=sum+list_pred[j][i]
            if sum>=5:
                pred.append(1)
            else:
                pred.append(0)
        return pred

