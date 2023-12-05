import numpy as np
import math
from randomfores_class.decisiontree import DecisionTreeClassifier, get_majority_class


class RandomForestClassifier():
    def __init__(self, n_estimators=2, max_depth=5, bootstrap_fraction=0.5, features_fraction=0.5):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.bootstrap_fraction = bootstrap_fraction
        self.features_fraction = features_fraction
        self.estimators = []

    def fit(self, x, y):
        num_rows = math.ceil(self.bootstrap_fraction * x.shape[0])
        num_cols = math.ceil(self.features_fraction * x.shape[1])
        for _ in range(self.n_estimators):
            #
            # Create noisy subsets x_subset, y_subset here
            rows_idx = np.random.choice(x.shape[0], size = num_rows)
            cols_idx = np.random.choice(x.shape[1], size = num_cols, replace = False)
            x_subset = x[rows_idx][:, cols_idx]
            y_subset = y[rows_idx]

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(x_subset, y_subset)
            self.estimators.append((tree,cols_idx))

    def predict(self, x):
        allpreds = np.array([e.predict(x[:,cols]) for e, cols in self.estimators])
        predictions = np.array([get_majority_class(y) for y in allpreds.T])
        return predictions
