from collections import namedtuple
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

Leaf = namedtuple('Leaf', ('value', 'x', 'y'))
Node = namedtuple('Node', ('feature', 'value', 'impurity', 'left', 'right',))


class BaseDecisionTree:
    def __init__(self, x, y, max_depth=np.inf):
        self.x = np.atleast_2d(x)
        self.y = np.atleast_1d(y)
        self.max_depth = max_depth

        self.features = x.shape[1]

        self.root = self.build_tree(self.x, self.y)


    def build_tree(self, x, y, depth=1):
        if depth > self.max_depth or self.criteria(y) < 1e-6:
            return Leaf(self.leaf_value(y), x, y)

        feature, value, impurity = self.find_best_split(x, y)

        left_xy, right_xy = self.partition(x, y, feature, value)
        left = self.build_tree(*left_xy, depth=depth + 1)
        right = self.build_tree(*right_xy, depth=depth + 1)

        return Node(feature, value, impurity, left, right)

    def partition(self, x, y, feature, value):
        i = x[:, feature] >= value
        j = np.logical_not(i)
        return (x[j], y[j]), (x[i], y[i])

    def _impurity_partition(self, value, feature, x, y):
        (_, left), (_, right) = self.partition(x, y, feature, value)
        return self.impurity(left, right)

    def find_best_split(self, x, y):
        best_feature, best_value, best_impurity = 0, x[0, 0], np.inf
        for feature in range(self.features):
            if x.shape[0] > 2:
                x_interval = np.sort(x[:, feature])
                res = optimize.minimize_scalar(
                    self._impurity_partition,
                    args=(feature, x, y),
                    bounds=(x_interval[1], x_interval[-1]),
                    method='Bounded',
                )
                assert res.success
                value = res.x
                impurity = res.fun
            else:
                value = np.max(x[:, feature])
                impurity = self._impurity_partition(value, feature, x, y)
            if impurity < best_impurity:
                best_feature, best_value, best_impurity = feature, value, impurity
        return best_feature, best_value, best_impurity


    def impurity(self, left, right):
        h_l = self.criteria(left)
        h_r = self.criteria(right)
        return (left.size * h_l + right.size * h_r) / (left.size + right.size)

    def leaf_value(self, y):
         return np.mean(y)

    def criteria(self, y):
        return np.sum((np.mean(y) - y)**2)

    def predict(self, x):
        x = np.atleast_2d(x)
        y = np.empty(x.shape[0], dtype=self.y.dtype)
        for i, row in enumerate(x):
            node = self.root
            while not isinstance(node, Leaf):
                if row[node.feature] >= node.value:
                    node = node.right
                else:
                    node = node.left
            y[i] = node.value
        return y
