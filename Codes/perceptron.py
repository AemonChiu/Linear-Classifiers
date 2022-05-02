"""Perceptron model."""

import numpy as np

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def get_accuracy(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        N, D = X_train.shape
        self.w = np.random.rand(D, self.n_class) # D rows, no_of_classes columns
        for epoch in range(self.epochs):
            print("epoch " + str(epoch + 1))
            for i in range(N):
                class_label = y_train[i] # the label indicated by y for the i-th item
                pred = self.w.T.dot(X_train[i]) # the value caculated by the dot product of w.T and X for the i-th item
                class_pred = np.argmax(pred) # the label caculated by the dot product of w.T and X for the i-th item
                if class_pred == class_label:
                    continue # if the label is correct, do nothing
                for c in range(self.n_class): # if the label is wrong
                    if self.w.T[c].dot(X_train[i]) > self.w.T[y_train[i]].dot(X_train[i]): # if w_c.T > w_yi.T * x_i
                        self.w.T[y_train[i]] += self.lr * X_train[i] # w_yi <- w_yi + lr * x_i
                        self.w.T[c] -= self.lr * X_train[i] # w_c <- w_c - lr * x_i
            print('The training accuracy is: %f' % self.get_accuracy(self.predict(X_train), y_train))
            self.lr = self.lr * 0.9 # learning rate decay
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
            predict = np.argmax(self.w.T.dot(test))
            output.append(predict)
        return output
