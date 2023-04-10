# Implementation of Non-Linear Decision Trees by [Ittner et al.]

import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator

from OC1Regression import OC1_Regression


class NonLinear_DTRegr(BaseEstimator):

    def __init__(self, criterion, max_depth, min_samples_split=1):

        self.criterion = criterion                     # splitting criterion
        self.max_depth = max_depth                     # maximum depth of the tree
        self.min_samples_split = min_samples_split     # number of samples to consider when looking for the best split
        self.Regr = OC1_Regression(criterion=self.criterion, max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split)   # OC1 Regressor by [Murthy et al.]

    def fit(self, X, y):
        X = X
        y = y
        X = NewFeatureSpace(X)
        self.Regr.fit(X, y)

    def predict(self, X):
        X = NewFeatureSpace(X)
        y = self.Regr.predict(X)
        return y

# Generating new features for the oblique decision tree
def NewFeatureSpace(X):

    comb = combinations(np.arange(len(X[0, :])), 2)
    new_X = X
    for i in list(comb):
        new_feature = X[:,i[0]]*X[:,i[1]]
        new_feature = np.array(new_feature)
        new_X =np.column_stack((new_X, new_feature))

    for i in range(len(X[0, :])):
        new_feature = X[:, i] * X[:, i]
        new_feature = np.array(new_feature)
        new_X = np.column_stack((new_X, new_feature))

    return new_X
