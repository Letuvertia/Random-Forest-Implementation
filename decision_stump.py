import random
import numpy as np


class BranchingDecisionStump(object):
    def __init__(self):
        self.log = {}
        self.best_param = None
        self.best_impurity = float('inf')

    def branch(self, x, y, feature_idx):
        assert x.shape == y.shape
        data_size = x.shape[0]

        # sorting
        sorted_idx = np.argsort(x)
        x, y = x[sorted_idx], y[sorted_idx]

        if feature_idx not in self.log:
            self.log[feature_idx] = {}

        theta_ls = [float('-inf')]
        for idx in range(data_size-1):
            theta_ls.append((x[idx]+x[idx+1])/2)
        theta_ls.append(float('inf'))

        for idx, theta in enumerate(theta_ls):
            impurity = ((idx) * self.gini_index(y[:idx]) + 
                        (data_size-idx) * self.gini_index(y[idx:]) )
            
            self.log[feature_idx][theta] = impurity
            if impurity < self.best_impurity:
                self.best_impurity = impurity
                self.best_param = (feature_idx, theta, idx)
            if impurity == self.best_impurity:
                if (feature_idx < self.best_param[0] or 
                    feature_idx == self.best_param[0] and theta < self.best_param[1]):
                    self.best_impurity = impurity
                    self.best_param = (feature_idx, theta, idx)
    

    @staticmethod
    def gini_index(array):
        """ return the impurity function of gini-index

        Input:
            :array: 1-D numpy array
        """
        (unique, counts) = np.unique(array, return_counts=True)
        return 1 - np.sum((counts / array.shape[0])**2)


    def fit(self, data_x, data_y, verbose=True):
        """ find the best branching criteria

        Input:
            :data_x: 2-D numpy array
                size = (data set size, the number of features)
            :data_y: 1-D numpy array
                size = (data set size,)
        """
        for feature_idx in range(data_x.shape[1]):
            self.branch(data_x[:, feature_idx], data_y, feature_idx)
        
        data_size = data_x.shape[0]
        feature_idx, theta, idx = self.best_param

        sorted_idx = np.argsort(data_x[:, feature_idx])
        sorted_x, sorted_y = data_x[sorted_idx], data_y[sorted_idx]

        if verbose:
            showing_idx = min(idx, data_size-idx)
            print("The fitting data size: {}; branch into {} and {}".format(data_x.shape, showing_idx, data_size-showing_idx))
            print("The best split param: feature_idx:{}; theta:{}".format(feature_idx, theta))

        return (sorted_x[:idx], sorted_y[:idx], sorted_x[idx:], sorted_y[idx:])
    

    def predict(self, data_x):
        """
        Input:
            :data_x: 1-D numpy array
                size = (the number of features,)
        """
        feature_idx, theta, _ = self.best_param
        if data_x[feature_idx] <= theta:
            return 1
        elif data_x[feature_idx] > theta:
            return 2