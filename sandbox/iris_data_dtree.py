##
import numpy as np 
from sklearn.datasets import load_iris
import pandas as pd 


def gini_index(y):
    classes, counts = np.unique(y, return_counts=True)
    gini = 1.0 - sum((count / len(y)) ** 2 for count in counts)
    return gini


def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(p * np.log2(p) for p in probs if p > 0)


def split_tree(df, feature_name, split_value):
    left_node = df[df[feature_name] <= split_value]
    right_node = df[df[feature_name] > split_value]
    
    return left_node, right_node

X =  load_iris().data
y = load_iris().target


X_new,y_new = np.zeros((20,4)), np.zeros(20)

X_new[0:5], y_new[0:5] = X[0:5], y[0:5]
X_new[5:10], y_new[5:10] = X[65:70], y[65:70]
X_new[10:15], y_new[10:15] = X[115:120], y[115:120]
X_new[15:20], y_new[15:29] = X[145:150], y[145:150]

del X,y

X = X_new.copy()
y = y_new.copy()

df = pd.DataFrame(X)
df['y'] = y

d1_left_tree, d1_right_tree = split_tree(df, 0, 5.6)


d21_left_tree, d21_right_tree = split_tree(d1_left_tree, 1, 3.0)
d22_left_tree, d22_right_tree = split_tree(d1_right_tree, 1, 3.0)

# np.save("../datasets/dtree_iris_X.npy",X)
# np.save("../datasets/dtree_iris_y.npy",y)

# pd.DataFrame(X).to_csv("../datasets/dtree_iris_X.csv", index=False)
# pd.DataFrame(y).to_csv("../datasets/dtree_iris_y.csv", index=False )


# np.unique(X,return_vales=True)
