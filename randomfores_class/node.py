import numpy as np

def compute_entropy(labels):
    # This function should compute the entropy
    #= sum_l(-p_l log2 (p_l)) for each label l)
    # of the input array.
    x = []
    un_labels = np.unique(labels)
    for l in un_labels:
        p = len(np.where(labels == l)[0])/len(labels)
        ent = -p*np.log2(p)
        x.append(ent)
        
    return np.sum(x)

class Node():
    def __init__(self):
        self.threshold = None
        self.col = None
        self.is_leaf = None
        self.output_class = None
        self.left_child = None
        self.right_child = None

    def find_best_question(self, x, y):
        # x: np array of shape (number of examples, number of features)
        # y: np array of shape (number of examples,)
        best_col = 0
        best_val = 0
        best_loss = np.inf

        num_cols = x.shape[1]
        valid_cols = np.arange(num_cols)
        for col in valid_cols:
            # Compute the midpoints of this column's values here
            sorted_indices = np.argsort(x[:,col])
            sorted_vals = x[sorted_indices, col]
            midpoints = []
            for i in range(len(sorted_vals)-1):
                midpoints.append((sorted_vals[i]+sorted_vals[i+1])/2)

            for val in midpoints:
                # Using col and val, split the labels
                # into left_labels, right_labels here

                right_subset_rows = x[:,col] > val
                left_subset_rows = x[:,col] <= val

                right_labels = y[right_subset_rows]
                left_labels = y[left_subset_rows]

                right_entropy = compute_entropy(right_labels)
                left_entropy = compute_entropy(left_labels)

                p_right = len(right_labels)/len(y)
                p_left = len(left_labels)/len(y)


                loss =  p_left*left_entropy + p_right*right_entropy


                if right_labels.shape[0] == 0 or left_labels.shape[0] == 0:
                    continue

                if loss < best_loss:
                    best_loss = loss
                    best_col = col
                    best_val = val

        self.col = best_col
        self.threshold = best_val

    def ask_question(self, x):
        if not self.is_leaf:
            return x[:, self.col] > self.threshold
        else:
            print("Error: leaf nodes cannot ask questions!")
            return False

    def predict(self):
        if self.is_leaf:
            return self.output_class
        else:
            print("Error: non-leaf nodes cannot make a prediction!")
            return None
