from sklearn import svm

from decision_regression_trees import create_trees_testing_dataframe
from utils import *


def test_svm(movies_metadata_dataframe, credits_dataframe):
    df = create_trees_testing_dataframe(movies_metadata_dataframe, credits_dataframe)

    X = df.drop(columns="is_successful")
    y = df["is_successful"]

    test_svc_with_cv(X, y, cls=svm.SVC)


def test_svc_with_cv(X, y, cls):
    """ Tests SVM with some some training
        (X_train, y_train) and testing (X_test, y_test) data
    """
    print("\nTesting {} with Radial Basis Function (RBF) as kernel".format(cls.__name__))
    # Radial Basis Function (RBF)
    classifier = cls()
    show_cross_validation_score(classifier, X, y)
