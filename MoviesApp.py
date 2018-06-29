import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from test_functions import *
from preprocessing import *
from utils import *
from naive_bayes import *
from svm_classifier import *
from decision_regression_trees import *
from instance_based import *
from features_improement import *

def test_algorithms(movies_metadata_dataframe, credits_dataframe):
    """ Tests several (decision tree) algorithms usign some dataframes """
    # Creates testing dataframe
    df = create_trees_testing_dataframe(movies_metadata_dataframe, credits_dataframe)

    # Tests decision tree or gradient boosting classification
    print("Predicting the success of the movies using classification ...")
    X = df.drop(columns="is_successful")
    y = df["is_successful"]

    test_decision_tree_classification_with_cv(X, y)
    test_gradient_boosting_classification_with_cv(X, y)
    test_knn_classification_with_cv(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=0.33,
                                                       random_state=42)

    test_gradient_boosting_classification(X_train, X_test, y_train, y_test)
    test_decision_tree_classification(X_train, X_test, y_train, y_test)

    # Tests regression tree, liear regression and boosting
    print("\nPredicting revenue using regression ...")
    print("  The shape of the dataframe before filtering is: ", df.shape)
    df["revenue"] = movies_metadata_dataframe["revenue"]
    print("  The revenue values equal to 0 are: ", df[df["revenue"] == 0.0]["revenue"].count())
    print("  Replacing the revenue's 0 values with NaN ...")
    df["revenue"].replace(0.0, np.nan, inplace=True)
    print("  The revenue values that are NaN are: ", df[df['revenue'].isnull()].shape)
    print("  Filtering the datafram by the revenue values that are not NaN ...")
    df = df[df['revenue'].notnull()]
    print("  The shape of the dataframe after filtering is: ", df.shape)
    show_columns_with_nan(df)
    #X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="revenue"),
    #                                                   df["revenue"],
    #                                                   test_size=0.33,
    #                                                   random_state=42)
    X = df.drop(columns="revenue")
    y = df["revenue"]
    test_linear_regression_with_cv(X, y)
    test_decision_tree_regression_with_cv(X, y)
    test_gradient_boosting_regression_with_cv(X, y)
    test_knn_regression_with_cv(X, y)


def main():
    movies_metadata_test_file_path = "movies_metadata_test.csv"
    movies_metadata_file_path = "files/the-movies-dataset/movies_metadata.csv"
    credits_file_path = "files/the-movies-dataset/credits.csv"
    ratings_file_path = "files/the-movies-dataset/ratings_small.csv"
    imdb_movies_file_path = "files/imdb/imdb.csv"

    print("Reading movies data ...")

    # Reads the movies metadata data
    print("  Reading the movies' metadata ...")
    movies_metadata_dataframe = read_data(movies_metadata_file_path)

    # Reads the ratings data
    print("  Reading the movies' ratings ...")
    ratings_dataframe = read_data(ratings_file_path)

    # Reads the credits data
    print("  Reading the movies' credits ...")
    credits_dataframe = read_data(credits_file_path)

    # Reads the imdb movies data
    #print("  Reading the imdb movies data ...")
    #imdb_movies_dataframe = read_data(imdb_movies_file_path)

    # Adds the the imdb movies data to the imds_movies_dataframe
    #movies_metadata_dataframe = add_new_columns_from_imdb_movies_dataframe(imdb_movies_dataframe,
    #                                                                      movies_metadata_dataframe)

    # Preprocesses the movies' metadata
    movies_metadata_dataframe = preprocess_movies_metadata(movies_metadata_dataframe, False)

    # Preprocesses the movies' credits
    credits_dataframe = preprocess_movies_credits(credits_dataframe)

    # Tests algorithms with some data
    test_algorithms(movies_metadata_dataframe, credits_dataframe)
    naive_bayes(movies_metadata_dataframe, credits_dataframe)
    test_svm(movies_metadata_dataframe, credits_dataframe)


if __name__ == "__main__":
    # Enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
