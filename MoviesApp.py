import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
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
    raw_cast_data = credits_dataframe[credits_dataframe["id"] == int(movie_id)]["cast"]
    cast_data = ast.literal_eval(raw_cast_data.iloc[0])
    actors_data = {actor["id"]: actor["name"] for actor in cast_data}
    #print(cast_data)
    if not cast_data:
        director_data = {0:""}
    return actors_data

def get_directors_data_by_movie_id(credits_dataframe, movie_id):
    raw_crew_data = credits_dataframe[credits_dataframe["id"] == int(movie_id)]["crew"]
    crew_data = ast.literal_eval(raw_crew_data.iloc[0])
    director_data = {crew["id"]: crew["name"] for crew in crew_data if crew["job"] == "Director"}
    #print(crew_data)
    if not crew_data:
        director_data = {0:""}
    return director_data

def read_data(file_path):
    dataframe = pd.read_csv(file_path, low_memory=False)
    return dataframe

def test_decission_tree(X_train, X_test, y_train, y_test):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
    max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    result = accuracy_score(y_test, y_pred)
    print(result)

def test_decission_tree_regression(X_train, X_test, y_train, y_test):
    clf_entropy = DecisionTreeRegressor(max_depth=2)
    clf_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    #result = accuracy_score(y_test, y_pred)
    #print(result)

def get_categorical_data_encoder(data):
    le = LabelEncoder()
    le.fit(data)
    #fitted_tittle = le.transform(movies_metadata_dataframe["title"][0:4])
    #list(le.inverse_transform([2, 2, 1]))
    return le

def get_all_actors_data(credits_dataframe, movies_metadata_dataframe):
    movies_metadata_dataframe_ids = movies_metadata_dataframe["id"].values
    all_actors_data = list()
    for id in movies_metadata_dataframe_ids:
        result = get_actors_data_by_movie_id(credits_dataframe, id)
        all_actors_data.append(result.values())
    return all_actors_data

def get_all_directors_data(credits_dataframe, movies_metadata_dataframe):
    movies_metadata_dataframe_ids = movies_metadata_dataframe["id"].values
    all_directors_data = list()
    for id in movies_metadata_dataframe_ids:
        result = get_directors_data_by_movie_id(credits_dataframe, id)
        all_directors_data.append(list(result.keys())[0]) 
    return all_directors_data

def get_one_hot_multilabled_actors_dataframe(all_actors_data):
    df = pd.DataFrame(columns=["actors"])
    for item in all_actors_data:
        df = df.append({"actors" :item}, ignore_index=True )
    
    mlb = MultiLabelBinarizer()
    test = df["actors"]
    mlb_result = mlb.fit_transform(test)
    df1 = pd.DataFrame(mlb_result, columns=mlb.classes_, index=df.index)
    print("One-hot mulilabled actors dataframe:")
    print(df1)
    print("\n")
    return df1

def edit_float_data(float_data):
    float_data_edited = []
    for item in float_data:
        if item.replace('.','', 1).isdigit():
            float_data_edited.append(float(item))
        else:
            item = sum(float_data_edited)/float(len(float_data_edited))
            float_data_edited.append(float(item))
    return float_data_edited

def test_decission_tree(credits_dataframe, movies_metadata_dataframe):
    all_popularity_data = movies_metadata_dataframe["popularity"].values
    #all_popularity_data.replace("he uses this to woo local beauty Beatrice.", 10.168437)
    #np.place(all_popularity_data, [str.isdecimal(item) for item in all_popularity_data], 10.168437)
    all_popularity_data_edited = edit_float_data(all_popularity_data)
    #all_popularity_data[57] = '10.168437'
    
    df1 = pd.DataFrame({"popularity": all_popularity_data_edited})
    vote_average = edit_float_data(movies_metadata_dataframe["vote_average"].values)
    df1["vote_average"] = pd.Series(vote_average)
    print(df1)
    
    all_directors_data = get_all_directors_data(credits_dataframe, 
                                                movies_metadata_dataframe)
    df1["director"] = pd.Series(all_directors_data)
    print(df1)
    
    X_train, X_test, y_train, y_test = train_test_split(df1.drop(columns="popularity"),
                                                       df1["popularity"],
                                                       test_size=0.33,
                                                       random_state=42)
    
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
    
    test_decission_tree_regression(X_train, X_test, y_train, y_test)

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
    movies_metadata_file_path = "../../movies/the-movies-dataset/movies_metadata.csv"
    credits_file_path = "../../movies/the-movies-dataset/credits.csv"
    ratings_file_path = "../../movies/the-movies-dataset/ratings_small.csv"

    # Read the movies metadata data
    print("Reading the movies' metadata ...")
    movies_metadata_dataframe = read_data(movies_metadata_test_file_path)

    # Read the ratings data
    print("Reading the movies' ratings ...")
    ratings_dataframe = read_data(ratings_file_path)

    # Read the credits data
    print("Reading the movies' credits ...")
    credits_dataframe = read_data(credits_file_path)

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
    #    get_one_hot_multilabled_actors_dataframe(all_actors_data)

    # Tests decission tree with some data
    #test_decission_tree(credits_dataframe, movies_metadata_dataframe)

    # Plots a dataframe
    #plot_dataframe(movies_metadata_dataframe, "vote_count")
    #ratings_dataframe[:100]["rating"].plot(kind='bar')    
    #plt.show()

if __name__ == "__main__":
    # Enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
