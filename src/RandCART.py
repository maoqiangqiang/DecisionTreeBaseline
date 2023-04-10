from sklearn.base import BaseEstimator
from scipy.linalg import qr
import numpy as np
from copy import deepcopy

class Node:

    def __init__(self, depth, labels, **kwargs):
        self.depth = depth
        self.labels = labels
        self.is_leaf = kwargs.get('is_leaf', False)
        self._split_rules = kwargs.get('split_rules', None)
        self._weights = kwargs.get('weights', None)
        self._left_child = kwargs.get('left_child', None)
        self._right_child = kwargs.get('right_child', None)

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
        if not hasattr(self, '_label'):
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


class Rand_CART(BaseEstimator):

    def __init__(self, impurity, segmentor, **kwargs):
        self.impurity = impurity
        self.segmentor = segmentor
        self._max_depth = kwargs.get('max_depth', None)
        self._min_samples = kwargs.get('min_samples', 1)
        self._compare_with_cart = kwargs.get('compare_with_cart', False)
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
            node = Node(cur_depth, y, is_leaf=True)
            self._nodes.append(node)
            return node
    
    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)
        else:
            n_objects, n_features = X.shape
            
            #generate random rotation matrix
            matrix = np.random.multivariate_normal(np.zeros(n_features), 
                                                  np.diag(np.ones((n_features))), 
                                                  n_features)
            Q, R = qr(matrix)
            X_rotation = X.dot(Q)
            
            impurity_rotation, sr_rotation, left_indices_rotation, right_indices_rotation = self.segmentor(X_rotation, y, self.impurity)
            
            if self._compare_with_cart:
                impurity_best, sr, left_indices, right_indices = self.segmentor(X, y, self.impurity)
                if impurity_best > impurity_rotation:
                    impurity_best = impurity_rotation
                    left_indices = left_indices_rotation
                    right_indices = right_indices_rotation
                    sr = sr_rotation
                else:
                    Q = np.diag(np.ones(n_features))
            else:
                impurity_best = impurity_rotation
                left_indices = left_indices_rotation
                right_indices = right_indices_rotation
                sr = sr_rotation
                
            if not sr:
                return self._generate_leaf_node(cur_depth, y)
            
            i, treshold = sr
            weights = np.zeros(n_features + 1)
            weights[:-1] = Q[:, i]
            weights[-1] = treshold
            
            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]
            
            ## add the part of min_samples into this script 
            if (len(y_right) <= self._min_samples):
                return self._generate_leaf_node(cur_depth, y)
            elif (len(y_left) <= self._min_samples):
                return self._generate_leaf_node(cur_depth, y)
            else:
                node = Node(cur_depth, y,
                            split_rules=sr,
                            weights=weights,
                            left_child=self._generate_node(X_left, y_left, cur_depth + 1),
                            right_child=self._generate_node(X_right, y_right, cur_depth + 1),
                            is_leaf=False)
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
        predictions = np.empty((size, ), dtype=float)
        for i in range(size): 
            predictions[i] = predict_single(X[i, :])
        return predictions

    def score(self, data, labels):
        if not self._root:
            raise Exception("Decision tree has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / labels.shape[0]
