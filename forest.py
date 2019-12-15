from collections import namedtuple
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import tree
from tree import BaseDecisionTree
from sklearn.model_selection import ShuffleSplit



class RandomForest(BaseDecisionTree):
    def __init__(self, x, y, max_depth=np.inf):
        super().__init__(x, y, max_depth=np.inf)
        self.x = np.atleast_2d(x)
        self.y = np.atleast_1d(y)
        self.root = self.build_forest(self.x, self.y)


    def build_forest(self, x, y, tree_num=10):
        n_obj, n_feat = x.shape
        self.tree_num = tree_num
        sample_size = int(len(x)/tree_num)
        curr_x = np.random.randint(0, n_obj, size=n_obj)
        h = []
        for i in range (0, len(x) + 1, sample_size):
            z = curr_x[i:i + sample_size]
            h.append((x[z], y[z]))

        forest = []
        for list in h:
            forest.append(self.build_tree(list[0], list[1]))
        return forest


    def predict(self, x):
        result = []
        for j in range(self.tree_num):
            y = np.zeros(x.shape[0])
            for i, row in enumerate(x):
                node = self.root[j]
                while not isinstance(node, tree.Leaf):
                    if row[node.feature] >= node.value:
                        node = node.right
                    else:
                        node = node.left
                y[i] = node.value
            result.append(y)
        return np.mean(np.array(result), axis=0)
