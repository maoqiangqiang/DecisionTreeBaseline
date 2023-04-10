import numpy as np 
from sklearn.base import BaseEstimator
from scipy.stats import mode
from HHCART import MSE


## main function for OC1 regression
class OC1_Regression(BaseEstimator):

    def __init__(self, criterion, max_depth, min_samples_split=1):

        # Get the options for tree learning
        self.criterion = criterion                      # splitting criterion --- 'MSE()'
        self.max_depth = max_depth                      # maximum depth of the tree
        self.min_samples_split = min_samples_split      # minimum number of samples needed for a split
        self.tree_ = None                               # Internal tree --- initially set as 'None'

    def fit(self, X, y):
        X = X
        y = y
        # Ensure that X is a 2d array of shape (n_samples, n_features)
        if X.ndim == 1:                 # single feature from all examples
            if len(y) > 1:
                X = X.reshape(-1, 1)
            elif len(y) == 1:           # single training example
                X = X.resphape(1, -1)
            else:
                ValueError('Invalid X and y')

        # fix the criterion here as MSE; but can be changed to other criterion
        if self.criterion == 'mse':
            self.criterion = MSE()
        else:
            Exception('Invalid criterion')

        # Build a tree and get its root node
        self.root_node, self.learned_depth = build_oblique_tree_oc1(X, y, self.criterion, self.max_depth, self.min_samples_split)
        
        # Create a tree object
        self.tree_ = Tree()
        self.tree_.set_root_node(self.root_node)
        self.tree_.set_depth(self.learned_depth)

    def predict(self, X):
        y = self.tree_.root_node.predict(X)
        return y

    # def get_params(self, deep = False):
    #     return {'max_depth': self.max_depth,
    #             'min_samples_split': self.min_samples_split,
    #             'criterion': self.criterion, 'min_features_split': self.min_features_split}

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self

class Tree:
    def __init__(self): # Defining parameters for a single node.
        self.root_node = None                      #  Initially 'root_node' will be set to 'None'.
        self.depth = -1                            #  Depth of the node.
        self.num_leaf_nodes = -1                   #  Number of leaf nodes in the internal nodes.

    def set_root_node(self, root_node):
        self.root_node = root_node

    def set_depth(self, depth):
        self.depth = depth

    def get_depth(self):
        if self.depth == -1:
            NotImplementedError('TODO: depth first traversal')
        return self.depth

    def predict(self, X):
        return self.root_node.predict(X)
    

class Node:
    def __init__(self, w, b, value=None, conf=0.0, samples=None, features=None):
        self.w = w                  # weights
        self.b = b                  # bias
        self.value = value          # value of the current node if the node is a leaf
        self.conf = conf            # score of the node, typically accuracy or r2
        self.samples = samples      # training examples in this node
        self.features = features    # features used in this node
        self.left_child = None
        self.right_child = None
        self.is_fitted = False      # flag to check if the node has been fitted

    def add_left_child(self, child):
        self.left_child = child

    def add_right_child(self, child):
        self.right_child = child

    def is_leaf(self):
        return (self.left_child is None) and (self.right_child is None)

    def predict(self, X):

        # Partition the data based on the split
        y = (np.dot(X, self.w) + self.b).squeeze()
        left, right = (y <= 0), (y > 0)
        y[left] = self.left_child.predict(X[left, :])
        y[right] = self.right_child.predict(X[right, :])

        return y

class LeafNode(Node):
    def __init__(self,  value=None, conf=0.0, samples=None, features=None):
        super(LeafNode, self).__init__(w=None, b=None, value=value, conf=conf, samples=samples, features=features)

    def predict(self, X):

        # Simply return the leaf value
        return np.full((X.shape[0], ), self.value)
    


# Implements Murthy et al (1994)'s algorithm to learn an oblique decision tree via random perturbations
def build_oblique_tree_oc1(X, y,  criterion,
                           max_depth, min_samples_split,
                           current_depth=0, current_features=None, current_samples=None, debug=False):

    n_samples, n_features = X.shape

    # Initialize
    if current_depth == 0:
        current_features = np.arange(n_features)
        current_samples = np.arange(n_samples)

    # Score the current node
    std = np.std(y)
    label = np.mean(y)
    conf = np.sum((-std <= y) & (y <= std)) / n_samples  # Regression confidence: fraction of examples within 1 std

    # Check termination criteria, and return a leaf node if terminating
    if (current_depth == max_depth or            # max depth reached
            n_samples <= min_samples_split or    # not enough samples to split on
            conf >= 0.95):                       # node is very homogeneous

        return LeafNode(value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    # Otherwise, learn a decision node
    feature_splits = get_best_splits(X, y, criterion=criterion)         # Get the best split for each feature
    f = np.argmin(feature_splits[:, 1])                                 # Find the best feature to split on
    best_split_score = feature_splits[f, 1]                             # Save the score corresponding to the best split
    w, b = np.eye(1, n_features, f).squeeze(), -feature_splits[f, 0]    # Construct a (w, b) from the best split
                                                                        # X[f] <= s becomes 1. X[f] + 0. X[rest] - s <= 0


    stagnant = 0                                                        # Used to track stagnation probability (see below)
    for k in range(5):                                                  # Randomly attempt to perturb a feature
        m = np.random.randint(0, n_features)                            # Select a random feature (weight w[m]) to update
        idx = np.where(X[:,m] == 0)[0]
        if len(idx) != 0:
            X[idx,m] = 1e-6

        wNew = np.array(w)                                              # Initialize wNew to w
        margin = (np.dot(X, wNew) + b)                                  # Compute the signed margin of all training examples
        u = (X[:, m]*w[m] - margin) / X[:, m]                           # Compute the residual of all training examples

        possible_wm = np.convolve(np.sort(u), [0.5, 0.5])[1:-1]         # Generate a list of possible values for new w[m]
        scores = np.empty_like(possible_wm)
        best_wm, best_wm_score = 0, np.inf                              # Find the best value for w[m]
        i = 0
        for wm in possible_wm:
            wNew[m] = wm                                                # Try (w = [w0, ..., wm, ..., wd], b) as a split
            margin = (np.dot(X, wNew) + b)                              # Signed margin of examples using this split
            left, right = y[margin <= 0], y[margin > 0]                 # Partition of examples using this split
            wm_score = criterion(left, right)                           # Score of this split
            scores[i] = wm_score
            i += 1

            if wm_score < best_wm_score:                                # Save the best split
                best_wm_score = wm_score
                best_wm = wm

        # Once the best w[m] among possible u has been identified, check if its actually any good
        if best_wm_score < best_split_score:
            # If we've identified a split with a better score, update it
            best_split_score = best_wm_score
            w[m] = best_wm
            stagnant = 0
        elif np.abs(best_wm_score - best_split_score) < 1e-3:
            # If we've identified a split with a similar score, update it with probability P(update) = exp(-stagnant)
            # Stagnation prob. is the probability that the perturbation does not improve the score. To prevent the
            # impurity from remaining stagnant for a long time, the stag. prob decreases exponentially with the number
            # of stagnant perturbations. It is reset to 1 every time the global impurity measure is improved
            if np.random.rand() <= np.exp(-stagnant):
                best_split_score = best_wm_score
                w[m] = best_wm
                stagnant += 1

        # If we've achieved a fantastic split, stop immediately
        if best_split_score < 1e-3:
            print("break ...")
            break
    
    #....................Sanity Check.........................
    idx = np.where(w == np.inf)[0]
    if len(idx) != 0:
        w[idx]= np.random.rand(len(idx))
    idx = np.where(w == -np.inf)[0]
    if len(idx) != 0:
        w[idx] = -1 *(np.random.rand(len(idx)))
    if b == np.inf:
        print("b = inf")
        b = 10
    if b == -np.inf:
        print("b = -inf")
        b = -10

    # Now that we have a split, perform a final partition
    margin = np.dot(X, w) + b
    left, right = margin <= 0, margin > 0
    if (len(y[left]) == 0 ):
        return LeafNode( value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    elif(len(y[right]) == 0):
        return LeafNode(value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    else:
        # Create a decision node
        decision_node = Node(w, b,  value=label, conf=conf,
                               samples=current_samples, features=current_features)

        # Grow the left branch and insert it
        left_node, left_depth = build_oblique_tree_oc1(X[left, :], y[left], criterion,
                                                   max_depth, min_samples_split,
                                                   current_depth=current_depth + 1,
                                                   current_features=current_features,
                                                   current_samples=current_samples[left])
        decision_node.add_left_child(left_node)

        # Grow the right branch and insert it
        right_node, right_depth = build_oblique_tree_oc1(X[right, :], y[right],
                                                     criterion,
                                                     max_depth, min_samples_split,
                                                     current_depth=current_depth + 1,
                                                     current_features=current_features,
                                                     current_samples=current_samples[right])
        decision_node.add_right_child(right_node)

        return decision_node, max(left_depth, right_depth)




# Get the best splitting threshold for each feature/attribute by considering them independently of the others
def get_best_splits(X, y, criterion):
    n_samples, n_features = X.shape
    all_splits = np.zeros((n_features, 2))

    for f in range(n_features):
        feature_values = np.sort(X[:, f])
        feature_splits = np.convolve(feature_values, [0.5, 0.5])[1:-1]
        scores = np.empty_like(feature_splits)
        best_split = None
        best_score = np.inf

        # Compute the scores
        for i, s in enumerate(feature_splits):
            left, right = y[X[:, f] <= s], y[X[:, f] > s]
            scores[i] = criterion(left, right)
            if scores[i] < best_score:
                best_score = scores[i]
                best_split = s

        all_splits[f, :] = [best_split, best_score]

    return all_splits




