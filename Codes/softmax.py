"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def get_accuracy(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me

        N, D = X_train.shape
        grad_w = np.zeros((D, self.n_class))
        for i in range(N):
            score = X_train[i].dot(self.w.T)
            score -= np.max(score)
            softmax = np.exp(score) / np.sum(np.exp(score))
            for j in range(self.n_class):
                if j == y_train[i]:
                    grad_w[:, j] += (softmax[j] - 1) * X_train[i]
                else:
                    grad_w[:, j] += softmax[j] * X_train[i]
        grad_w = (grad_w + self.reg_const * self.w.T) / N
        return grad_w.T


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        randomise = np.arange(len(y_train))
        X_train = X_train[randomise]
        y_train = y_train[randomise]
        N, D = X_train.shape
        self.w = 0.01 * np.random.randn(self.n_class, D)
        batch = 1200
        mini_iter = N // batch
        print(int(mini_iter))
        for epoch in range(self.epochs):
            print("epoch: " + str(epoch + 1))
            for i in range(int(mini_iter)):
                x = X_train[i * batch : (i+1) * batch]
                y = y_train[i * batch : (i+1) * batch]
                self.w -= self.lr * self.calc_gradient(x, y)
            print('The training accuracy is: %f' % self.get_accuracy(self.predict(X_train), y_train))
            self.lr *= 0.85
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
        output = []
        for test in X_test:
            output.append(np.argmax(self.w.dot(test)))
        return output
