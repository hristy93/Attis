import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from test_functions import *
from preprocessing import *
from utils import *
from naive_bayes import *
from decision_regression_trees import *


def test_algorithms(movies_metadata_dataframe, credits_dataframe):
    """ Tests several (decision tree) algorithms usign some dataframes """
    # Test1
    #all_popularity_data = movies_metadata_dataframe["popularity"].value   
    #df1 = pd.DataFrame({"popularity": all_popularity_data})

    #vote_average = movies_metadata_dataframe["vote_average"])
    #df1["vote_average"] = vote_average
    #print(df1)
    
    #all_directors_data = get_all_directors_data(credits_dataframe, 
    #                                            movies_metadata_dataframe)
    #df1["director"] = pd.Series(all_directors_data)
    #print(df1)

    #X_train, X_test, y_train, y_test = train_test_split(df1.drop(columns="popularity"),
    #                                                   df1["popularity"],
    #                                                   test_size=0.33,
    #                                                   random_state=42)


    # Test 2
    df = create_trees_testing_dataframe(movies_metadata_dataframe, credits_dataframe)

    # Test 2.1 - NOT RELEVANT
    #average = 68787389
    #df = df[df["revenue"] != average]
    #X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="revenue"),
    #                                                   df["revenue"],
    #                                                   test_size=0.33,
    #                                                   random_state=42)
    #test_decision_tree_regression(X_train, X_test, y_train, y_test)
    #test_gradient_boosting_regression(X_train, X_test, y_train, y_test)


    # Test 2.2 - Decision tree or gradient boosting classification
    print("Predicting the success of the movies using classification ...")
    X = df.drop(columns="is_successful")
    y = df["is_successful"]

    test_decision_tree_classification_with_cv(X, y)
    test_gradient_boosting_classification_with_cv(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=0.33,
                                                       random_state=42)

    test_gradient_boosting_classification(X_train, X_test, y_train, y_test)
    test_decision_tree_classification(X_train, X_test, y_train, y_test)

    # Test 2.3 - Regression tree, liear regression and boosting - NEEDS IMPROVEMENTS
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

    # Test 2.4 - Association rules - NOT WORKING (the frequest-items datagrame is empty)
    # The data should be one-hot !!!
    #support = 0.6
    #pd.set_option('max_columns', 10)
    #genres_one_hot = get_one_hot_multilabled_dataframe(movies_metadata_dataframe, "genres")
    #test_association_rules(genres_one_hot, support)
    

    # Test 3 - NOT RELEVANT
    #one_hot_multilabled_actors_dataframe["vote_average"] = pd.Series(vote_average)
    #one_hot_multilabled_actors_dataframe["popularity"] = all_popularity_data_edited
    #X_train, X_test, y_train, y_test = train_test_split(one_hot_multilabled_actors_dataframe.drop(columns="popularity"),
    #                                                   one_hot_multilabled_actors_dataframe["popularity"],
    #                                                   test_size=0.33,
    #                                                   random_state=42)
    
    #X_train, X_test, y_train, y_test = train_test_split(one_hot_multilabled_actors_dataframe,
    #                                                   np.asarray(all_popularity_data_edited),
    #                                                   test_size=0.33,
    #                                                   random_state=42)
    
    #test_decision_tree_classification(X_train, X_test, y_train, y_test)
    #test_decision_tree_regression(X_train, X_test, y_train, y_test)
    #test_gradient_boosting_regression(X_train, X_test, y_train, y_test)

    # Other old tests - NOT RELEVANT

    #df_actors = df.actors
    #res1 = df_actors.str.join('|').str
    #res2 = res1.get_dummies()
    #dummies = pd.get_dummies(df)
    #res = str(full_actors_data).join(sep='*').str.get_dummies(sep='*')
    #print(dummies)

    #mask = movies_metadata_dataframe["title"].isnull()
    #var = movies_metadata_dataframe["title"].fillna("")
    #test = movies_metadata_dataframe["vote_average"].values
    #print(var)

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

    #print(len(movies_metadata_dataframe["wins_count"]))
    #print(movies_metadata_dataframe["wins_count"].isnull().sum())

    # Preprocesses the movies' metadata
    movies_metadata_dataframe = preprocess_movies_metadata(movies_metadata_dataframe, False)

    # Preprocesses the movies' credits
    credits_dataframe = preprocess_movies_credits(credits_dataframe)

    # Filters the movie's metadata on the vote count
    #quantile = 0.75
    #remove_movies_with_less_votes(movies_metadata_dataframe, quantile)

    # Gets the actors data as dict of id to name
    #movie_id = 862
    #actors_data = get_actors_data_by_movie_id(credits_dataframe, movie_id)
     
    # Gets all actors data
    #all_actors_data = get_all_actors_data(credits_dataframe,
    #                                      movies_metadata_dataframe)

    # Gets the directors data as dict of id to name
    #movie_id = 862
    #director_data = get_directors_data_by_movie_id(credits_dataframe, movie_id)

    # Gets all directors data
    #all_directors_data = get_all_directors_data(credits_dataframe, 
    #                                            movies_metadata_dataframe)

    # Gets the ratings of a movie by id
    #movie_ratings = get_ratings_by_movie_id(ratings_dataframe, movie_id)

    # Gets a n-hot mulilabled prepresentation of the actors in the movies
    #one_hot_multilabled_actors_dataframe =\
    #    get_one_hot_multilabled_dataframe(all_actors_data, "actors")

    # Tests association rules - NOT WORKING
    #support = 0.6
    #association_rules_test(one_hot_multilabled_actors_dataframe, support)

    # Tests decision tree with some data
    test_algorithms(movies_metadata_dataframe, credits_dataframe)
    naive_bayes(movies_metadata_dataframe, credits_dataframe)

    # Plots a dataframe
    #plot_dataframe(movies_metadata_dataframe, "vote_count")
    #ratings_dataframe[:100]["rating"].plot(kind='bar')    
    #plt.show()


if __name__ == "__main__":
    # Enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
