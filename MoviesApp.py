import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import sys

def enable_win_unicode_console():
    try:
        # Fix UTF8 output issues on Windows console.
        # Does nothing if package is not installed
        from win_unicode_console import enable
        enable()
    except ImportError:
        pass

def plot_dataframe(dataframe, column_name):
    dataframe[column_name].plot()
    plt.show()

def get_ratings_by_movie_id(ratings_dataframe, movie_id):
    user_ratings = ratings_dataframe[ratings_dataframe["movieId"] == movie_id]['rating']
    return user_ratings

def get_actors_data_by_movie_id(credits_dataframe, movie_id):
    raw_cast_data = credits_dataframe[credits_dataframe["id"] == movie_id]["cast"]
    cast_data = ast.literal_eval(raw_cast_data[0])
    actors_data = {actor["id"]: actor["name"] for actor in cast_data}
    print(cast_data)

def read_data(file_path):
    dataframe = pd.read_csv(file_path, low_memory=False)
    return dataframe

def test_decission_tree(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
    max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)

def get_categorical_data_encoder(data):
    le = LabelEncoder()
    le.fit(data)
    #fitted_tittle = le.transform(movies_metadata_dataframe["title"][0:4])
    return le

def main():
    movies_metadata_test_file_path = "movies_metadata_test.csv"
    movies_metadata_file_path = "../../movies/the-movies-dataset/movies_metadata.csv"
    credits_file_path = "../../movies/the-movies-dataset/credits.csv"
    ratings_file_path = "../../movies/the-movies-dataset/ratings_small.csv"

    # Read the movies metadata data
    print("Read the movies' metadata ...")
    movies_metadata_dataframe = read_data(movies_metadata_test_file_path)

    # Read the ratings data
    print("Read the movies' ratings ...")
    ratings_dataframe = read_data(ratings_file_path)

    # Read the credits data
    print("Read the movies' credits ...")
    credits_dataframe = read_data(credits_file_path)

    # Gets the actors data as dict of id to name
    #movie_id = 862
    #actors_data = get_actors_data_by_movie_id(credits_dataframe, movie_id)

    # Plots a dataframe
    #plot_dataframe(movies_metadata_dataframe, "vote_count")
    #ratings_dataframe[:100]["rating"].plot(kind='bar')

    # Gets the ratings of a movie by id
    #movie_ratings = get_ratings_by_movie_id(ratings_dataframe, movie_id)

    #mask = movies_metadata_dataframe["title"].isnull()
    #var = movies_metadata_dataframe["title"].fillna("")
    #test = movies_metadata_dataframe["vote_average"].values
    #print(var)
    

if __name__ == "__main__":
    # enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
