from collections import Counter
from randomfores_class.node import Node
import numpy as np


def get_majority_class(labels):
    # This function should return the most common label
    # in the input array.
    return Counter(labels).most_common(1)[0][0]




class DecisionTreeClassifier():
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def create_node(self, x_subset, y_subset, depth):
        # Recursive function
        node = Node()

        majority_class = get_majority_class(y_subset)
        majority_class_count = (y_subset == majority_class).sum()
        perfectly_classified = majority_class_count == len(y_subset)

        if perfectly_classified or depth == self.max_depth:
            node.output_class = majority_class
            node.is_leaf = True
        else:
            node.find_best_question(x_subset,y_subset)
            node.is_leaf = False
            right_subset_rows = node.ask_question(x_subset)
            left_subset_rows = np.invert(right_subset_rows)
            # Recursion: create node.left_child and node.right_child here
            node.left_child = self.create_node(x_subset[left_subset_rows], y_subset[left_subset_rows], depth+1)
            node.right_child = self.create_node(x_subset[right_subset_rows], y_subset[right_subset_rows], depth+1)

        return node

    def fit(self, x, y):
        self.root_node = self.create_node(x,y,depth=1)

    def predict(self, x):
        predictions = []

        for i in range(len(x)):
            current_node = self.root_node
            x_i = x[i].reshape(1,-1)
            done_descending_tree = False
            while not done_descending_tree:
                if current_node.is_leaf:
                    predictions.append(current_node.predict())
                    done_descending_tree = True

                else:
                    if current_node.ask_question(x_i):
                        current_node = current_node.right_child
                    else:
                        current_node = current_node.left_child

        return np.array(predictions)