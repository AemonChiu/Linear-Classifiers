"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def get_accuracy(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1/(1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        N, D = X_train.shape
        self.w = np.random.randn(1, D)
        for epoch in range(self.epochs):
            print ("epoch: " + str(epoch + 1))
            for i in range(N):
                if y_train[i] == 1:
                    yi = 1
                else:
                    yi = -1
                self.w = self.w + self.lr * self.sigmoid(-yi * self.w.dot(X_train[i])) * yi * X_train[i]
            print('The training accuracy is: %f' % self.get_accuracy(self.predict(X_train), y_train))
            self.lr *= 0.9

        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N, D = X_test.shape
        output = np.empty(N)
        for test in range(N):
            if np.sign(self.w.dot(X_test[test])) == -1:
                output[test] = 0
            else:
                output[test] = 1
        return output
