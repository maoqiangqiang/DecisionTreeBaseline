from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
import numpy as np
from numpy.linalg import norm

class Node:
    
    """
        Decision node for RidgeCART.
    """
    
    def __init__(self, depth, labels, 
                 is_leaf=False, split_rules=None,
                method='usual', weights=None,
                left_child=None, right_child=None):
        self.depth = depth
        self.labels = labels
        self.is_leaf = is_leaf
        self._split_rules = split_rules
        self._method = method
        self._weights = weights
        self._left_child = left_child
        self._right_child = right_child

        if not self.is_leaf:
            assert self._split_rules
            assert self._left_child
            assert self._right_child

    def get_child(self, datum):
        
        """
            Get direction of input object in tree.
        """
        
        if self.is_leaf:
            raise Exception("Leaf node does not have children.")
        i, treshold = self._split_rules
        
        if self._method == 'usual':
            X_hat = np.zeros(datum.shape[0] + 1)
            X_hat[:-1] = datum
            X_hat[-1] = self._weights.dot(datum)
        elif self._method == 'hard':
            X_hat = np.array([self._weights.dot(datum)])

        if X_hat[i] < treshold:
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


class Ridge_CART(BaseEstimator):
    """
        RidgeCART is Oblique Decision Tree for regression tasks. Key idea of 
        this method is creating feature <x, w> where w - is coefficients of
        Ridge Regression without bias.
        
        Parameters
        -----------
        
        impurity : class of impurity. 
        Impurity is functional for optimization in decision tree nodes.
        
        segmentor : class of segmentor. 
        Find best cplit in node by impurity.
        
        alpha : float, optional (default=1.0). 
        Penalty parameter for Risge Regression.
        
        method : string, optional (default='usual')
        If method == 'usual' then feature <x, w> adding as new feature. 
        If method == 'hard' then feature <x, w> only one in feature space.
        
        max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples samples.
        
        min_samples : int, float, optional (default=2)
        The minimum number of samples required to split an internal node.
        
        
        Attribute
        ----------
        
        _root : Node class
        Root of decision tree.
        
        _nodes : list of nodes
        All nodes of decision tree.
    """
    def __init__(self, impurity, segmentor, 
                 alpha=1.0, method='usual', 
                max_depth=None, min_samples=2):
        self.impurity = impurity
        self.segmentor = segmentor
        self.alpha = alpha
        self.method = method
        self._max_depth = max_depth
        self._min_samples = min_samples
        self._root = None
        self._nodes = []
    
    def _terminate(self, X, y, cur_depth):
        """
            Terminate of building tree.
            
            
            Parameters
            -----------
            
            X : array-like, shape = [n_samples, n_features]
            The training set of this node.
            
            y : array-like, shape = [n_samples]
            The target values of this node.
            
            cur_depth : int
            Current depth.
            
            
            Return
            -----------
            
            terminate : bool
            If this node is leaf by max_depth or by min_samples then terminate = True.
        """
        if self._max_depth != None and cur_depth == self._max_depth:
            # maximum depth reached.
            return True
        elif y.size < self._min_samples:
            # minimum number of samples reached.
            return True
        else:
            return False
    
    def _generate_leaf_node(self, cur_depth, y):
        
        """
            Generate leaf node.
            
            Parameters
            -----------
            cur_depth : int
            Current depth.
            
            y : float
            Target of this leaf node.
        """
        
        node = Node(cur_depth, y, is_leaf=True)
        self._nodes.append(node)
        return node
    
    def _generate_node(self, X, y, cur_depth):
        """
            Generate current node.
            
            
            Parameters
            -----------
            
            X : array-like, shape = [n_samples, n_features]
            The training input samples for this node.
            
            y : array-like, shape = [n_samples]
            The target values for this node.
            
            cur_depth : int
            Current depth.
            
            Return
            
            node : object Node.
        """
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)
        else:
            n_objects, n_features = X.shape
            
            regressor = Ridge(alpha=self.alpha)
            regressor.fit(X, y)
            weights = regressor.coef_
            weights = weights / norm(weights)
            
            if self.method == 'usual':
                X_hat = np.hstack((X, X.dot(weights)[:, np.newaxis]))
            elif self.method == 'hard':
                X_hat = X.dot(weights)[:, np.newaxis]
            
            impurity, sr, left_indices, right_indices = self.segmentor(X_hat, y, self.impurity)
            
            if not sr:
                return self._generate_leaf_node(cur_depth, y)
            
            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]

            node = Node(cur_depth, y,
                        split_rules=sr,
                        method=self.method,
                        weights=weights,
                        left_child=self._generate_node(X_left, y_left, cur_depth + 1),
                        right_child=self._generate_node(X_right, y_right, cur_depth + 1),
                        is_leaf=False)
            self._nodes.append(node)
            return node
    
    def fit(self, X, y):
        """
            Build a decision tree regressor from the training set (X, y).
            
            
            Parameters
            -----------
            
            X : array-like, shape = [n_samples, n_features]
            The training input samples.
            y : array-like, shape = [n_samples]
            The target values.
            
            
            Return
            -----------
            
            self : object
            Returns self.
        """
        self._root = self._generate_node(X, y, 0)
        return self

    def predict(self, X):
        """
            Predict targets for testing set X.
            
            
            Parameters
            -----------
            
            X : array-like, shape = [n_samples, n_features]
            The testing input samples.
            
            
            Return
            -----------
            
            y : array-like, shape = [n_samples]
            The target values in test set.
        """
        
        def predict_single(x):
            """
                Predict fot single test object.
                
                
                Parameters
                -----------
                
                x : array-like, shape = [n_features]
                Test sample.
                
                Return
                -----------
                
                y : float
                Target of test sample.
            """
            cur_node = self._root
            while not cur_node.is_leaf:
                cur_node = cur_node.get_child(x)
            return cur_node.label
        
        if not self._root:
            raise Exception("Decision tree has not been trained.")
        size = X.shape[0]
        predictions = np.empty((size, ), dtype=float)
        for i in range(size): 
            predictions[i] = predict_single(X[i, :])
        return predictions

    def score(self, X, y):
        """
            Mean squared error regression score
        
        
            Parameters
            -----------
        
            X : array-like, shape = [n_samples, n_features]
            The training input samples.
            y : array-like, shape = [n_samples]
            The target values.
            
            Return
            ----------
            score : float
            Mean squared error.
        """
        if not self._root:
            raise ValueError("Decision tree has not been trained.")
        predictions = self.predict(X)
        return np.mean((y - predictions) ** 2)
