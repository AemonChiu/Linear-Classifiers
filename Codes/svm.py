"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        D, N = self.w.shape
        dW = np.zeros(self.w.shape)  # initialize the gradient as zero
        # compute the loss and the gradient
        num_classes = self.w.shape[1]
        num_train = X_train.shape[0]
        self.n_class = self.w.shape[1]
        # num_train = X_train.shape[0]
        loss = 0.0
        for i in range(num_train):
            scores = X_train[i].dot(self.w.T)
            correct_class_score = scores[y_train[i]]
            nb_sup_zero = 0
            for j in range(num_classes):
                if j == y_train[i]:
                    continue
                margin = scores - correct_class_score + 1  # note delta = 1
                if margin.all() > 0:
                    nb_sup_zero += 1
                    loss += margin
                    dW[:,j] += X_train[i]
                dW[:, y_train[i]] -= nb_sup_zero * X_train[i][j]
            loss /= num_train
            dW += self.lr * self.w
            loss += 0.5 * self.lr * np.sum(self.w * self.w)
            return loss, dW
        pass

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
        self.w = 0.01 * np.random.rand(self.n_class, D)
        batch = 1200
        mini_iter = N // batch
        print(int(mini_iter))
        for epoch in range(self.epochs):
            print("epoch " + str(epoch + 1))
            for i in range(int(mini_iter)):
                x = X_train[i * batch: (i + 1) * batch]
                y = y_train[i * batch: (i + 1) * batch]
                for j in range(batch):
                    xi = x[j]
                    yi = y[j]
                    for c in range(self.n_class):
                        self.w[c] = (1 - self.lr * self.reg_const / batch) * self.w[c]
                        if c != yi and self.w[yi].dot(xi) - self.w[c].dot(xi) < 1:
                            self.w[yi] = self.w[yi] + self.lr * xi
                            self.w[c] = self.w[c] - self.lr * xi
            print('The training accuracy is: %f' % self.get_accuracy(self.predict(X_train), y_train))
            self.lr *= 0.95
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
