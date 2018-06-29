import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from utils import *

def test_knn_classification_with_cv(X, y, k=30, algorithm='kd_tree', leaf_size=40):
    ''' algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional '''
    
    print("\nTesting KNeighborsClassifier {} ({} and {} leafs) with cross valdiation ...".format(k, algorithm, leaf_size))
    knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm=algorithm, leaf_size=leaf_size)
    show_cross_validation_score(knn_clf, X, y)

def test_knn_regression_with_cv(X, y, k=30, algorithm='kd_tree', leaf_size=40):
    ''' algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional '''
    
    print("\nTesting KNeighborsRegressor {} ({} and {} leafs) with cross valdiation ...".format(k, algorithm, leaf_size))
    knn_clf = KNeighborsRegressor(n_neighbors=k, weights='distance', algorithm=algorithm, leaf_size=leaf_size)
    show_cross_validation_score(knn_clf, X, y)
