from copy import deepcopy
import numpy as np
from scipy.linalg import norm
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


class HHCARTNode:
    def __init__(self, depth, labels, **kwargs):
        self.depth = depth
        self.labels = labels
        self.is_leaf = kwargs.get("is_leaf", False)
        self._split_rules = kwargs.get("split_rules", None)
        self._weights = kwargs.get("weights", None)
        self._left_child = kwargs.get("left_child", None)
        self._right_child = kwargs.get("right_child", None)

        if not self.is_leaf:
            assert self._split_rules
            assert self._left_child
            assert self._right_child

    def get_child(self, datum):
        if self.is_leaf:
            raise Exception("Leaf node does not have children.")
        X = deepcopy(datum)

        if X.dot(np.array(self._weights[:-1]).T) - self._weights[-1] < 0:
            return self.left_child
        else:
            return self.right_child

    @property
    def label(self):
        if not hasattr(self, "_label"):
            self._label = np.mean(self.labels)
        return self._label

    @property
    def split_rules(self):
        if self.is_leaf:
            raise Exception("Leaf node does not have split rule.")
        return self._split_rules

    @property
    def left_child(self):
        if self.is_leaf:
            raise Exception("Leaf node does not have split rule.")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise Exception("Leaf node does not have split rule.")
        return self._right_child


class HouseHolderCART(BaseEstimator):
    def __init__(self, impurity, segmentor, method="eig", tau=1e-4, **kwargs):
        self.impurity = impurity
        self.segmentor = segmentor
        self.method = method
        self.tau = tau
        self._max_depth = kwargs.get("max_depth", None)
        self._min_samples = kwargs.get("min_samples", 1)
        self._alpha = kwargs.get("alpha", None)  # only for linreg method
        self._root = None
        self._nodes = []

    def _terminate(self, X, y, cur_depth):
        if self._max_depth != None and cur_depth == self._max_depth:
            # maximum depth reached.
            return True
        elif y.size < self._min_samples:
            # minimum number of samples reached.
            return True
        elif np.unique(y).size == 1:
            return True
        else:
            return False

    def _generate_leaf_node(self, cur_depth, y):
        node = HHCARTNode(cur_depth, y, is_leaf=True)
        self._nodes.append(node)
        return node

    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)
        else:
            n_objects, n_features = X.shape
            impurity_best, sr, left_indices, right_indices = self.segmentor(
                X, y, self.impurity
            )
            oblique_split = False

            # generate Housholder matrix
            if self.method == "eig":
                extractor = PCA(n_components=1)
                extractor.fit(X, y)
                mu = extractor.components_[0]
            if self.method == "ridge":
                reg_model = Ridge(alpha=self._alpha)
                reg_model.fit(X, y)
                mu = reg_model.coef_

            I = np.diag(np.ones(n_features))
            check_ = np.sqrt(((I - mu) ** 2).sum(axis=1))
            if (check_ > self.tau).sum() > 0:
                i = np.argmax(check_)
                e = np.zeros(n_features)
                e[i] = 1.0
                w = (e - mu) / norm(e - mu)
                householder_matrix = I - 2 * w[:, np.newaxis].dot(
                    w[:, np.newaxis].T
                )
                X_house = X.dot(householder_matrix)
                (
                    impurity_house,
                    sr_house,
                    left_indices_house,
                    right_indices_house,
                ) = self.segmentor(X_house, y, self.impurity)

                if impurity_best > impurity_house:
                    oblique_split = True
                    impurity_best = impurity_house
                    left_indices = left_indices_house
                    right_indices = right_indices_house
                    sr = sr_house
            else:
                householder_matrix = I

            if not sr:
                return self._generate_leaf_node(cur_depth, y)

            i, treshold = sr
            weights = np.zeros(n_features + 1)
            if oblique_split:
                weights[:-1] = householder_matrix[:, i]
            else:
                weights[i] = 1
            weights[-1] = treshold

            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]

            ## add the part of min_samples into this script 
            if (len(y_right) <= self._min_samples):
                return self._generate_leaf_node(cur_depth, y)
            elif (len(y_left) <= self._min_samples):
                return self._generate_leaf_node(cur_depth, y)
            else:
                node = HHCARTNode(
                    cur_depth,
                    y,
                    split_rules=sr,
                    weights=weights,
                    left_child=self._generate_node(X_left, y_left, cur_depth + 1),
                    right_child=self._generate_node(
                        X_right, y_right, cur_depth + 1
                    ),
                    is_leaf=False,
                )
                self._nodes.append(node)
            return node

    def fit(self, X, y):
        self._root = self._generate_node(X, y, 0)

    def predict(self, X):
        def predict_single(datum):
            cur_node = self._root
            while not cur_node.is_leaf:
                cur_node = cur_node.get_child(datum)
            return cur_node.label

        if not self._root:
            raise Exception("Decision tree has not been trained.")
        size = X.shape[0]
        predictions = np.empty((size,), dtype=float)
        for i in range(size):
            predictions[i] = predict_single(X[i, :])
        return predictions

    def score(self, data, labels):
        if not self._root:
            raise Exception("Decision tree has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / labels.shape[0]






## segmentor.py
from abc import ABCMeta, abstractmethod
import numpy as np


class MSE:
    """
    Mean squared error impurity criterion
    """

    def __call__(self, left_label, right_label):
        left_len, right_len = len(left_label), len(right_label)

        left_std = np.std(left_label)
        right_std = np.std(right_label)

        total = left_len + right_len

        return (left_len / total) * left_std + (right_len / total) * right_std


class SegmentorBase:
    """
    Abstract segmentor class. Segmentor called in nodes for find best split.


    Parameters
    -----------

    msl : int, optional (default=1)
    The minimum number of samples required to be at a leaf nodes.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def _split_generator(self, X):
        """
        Abstract method for split.


        Parameter
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.
        """
        pass

    def __init__(self, msl=1):
        self._min_samples_leaf = msl

    def __call__(self, X, y, impurity=MSE()):
        """
        Parameters
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.
        y : array-like, shape = [n_samples]
        The target values.
        impurity : object of impurity (default=MSE()).
        The name of criterion to measure the quality of a split.


        Returns
        -----------

        Tuple of the following elements:
        best_impurity : float.
        The best value of impurity.
        best_split_rule : tuple.
        The pair of feature and value.
        best_left_i : numpy.ndarray.
        The indexes of left node objects.
        best_right_i : numpy.ndarray.
        The indexes of right node objects.
        """
        best_impurity = float("inf")
        best_split_rule = None
        best_left_i = None
        best_right_i = None
        splits = self._split_generator(X)

        for left_i, right_i, split_rule in splits:
            if (
                left_i.size >= self._min_samples_leaf
                and right_i.size >= self._min_samples_leaf
            ):
                left_labels, right_labels = y[left_i], y[right_i]
                cur_impurity = impurity(left_labels, right_labels)
                if cur_impurity < best_impurity:
                    best_impurity = cur_impurity
                    best_split_rule = split_rule
                    best_left_i = left_i
                    best_right_i = right_i
        return (best_impurity, best_split_rule, best_left_i, best_right_i)


class MeanSegmentor(SegmentorBase):
    """
    Split based on mean value of each feature.
    """

    def _split_generator(self, X):
        """
        Parameters
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.


        Returns
        -----------

        Tuple of the following elements:
        left_i : numpy.ndarray.
        The indexes of left node objects.
        right_i : numpy.ndarray.
        The indexes of right node objects.
        split_rule : tuple.
        The pair of feature and value.
        """
        for feature_i in range(X.shape[1]):
            feature_values = X[:, feature_i]
            mean = np.mean(feature_values)
            left_i = np.nonzero(feature_values < mean)[0]
            right_i = np.nonzero(feature_values >= mean)[0]
            split_rule = (feature_i, mean)
            yield (left_i, right_i, split_rule)
