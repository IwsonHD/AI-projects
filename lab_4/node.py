import copy
from random import sample
import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        for idx in possible_splits:
            left_classes = y[:idx + 1].astype(int)
            right_classes = y[idx + 1:].astype(int)
            
            left_counts = np.bincount(left_classes)
            right_counts = np.bincount(right_classes)
            
            left_size = len(left_classes)
            right_size = len(right_classes)
            total_size = left_size + right_size
            
            gini_left = 1 - sum((np.float64(count) / left_size) ** 2 for count in left_counts)
            gini_right = 1 - sum((np.float64(count) / right_size) ** 2 for count in right_counts)
            
            gini = (left_size / total_size) * gini_left + (right_size / total_size) * gini_right
            
            gain = -gini 
            
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        return best_idx, best_gain
    
    def gini_best_score_v2(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        for idx in possible_splits:
            left_classes = y[:idx + 1]
            right_classes = y[idx + 1:]
            
            left_pos = np.count_nonzero(left_classes == 1)
            left_neg = np.count_nonzero(left_classes == 0)
            right_pos = np.count_nonzero(right_classes == 1)
            right_neg = np.count_nonzero(right_classes == 0)
            
            left_size = len(left_classes)
            right_size = len(right_classes)
            total_size = left_size + right_size
            
            gini_left = 1 - ((left_pos / left_size) ** 2 + (left_neg / left_size) ** 2)
            gini_right = 1 - ((right_pos / right_size) ** 2 + (right_neg / right_size) ** 2)
            
            gini = (left_size / total_size) * gini_left + (right_size / total_size) * gini_right
            
            gain = gini_left + gini_right - gini
            
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        return best_idx, best_gain


    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None
        features = range(X.shape[1])

        # TODO implement feature selection
        if feature_subset is not None:
            features = sample(features, k=feature_subset)

        for d in features:
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
