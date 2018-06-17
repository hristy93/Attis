import ast
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import sys

from test_functions import *;
from preprocessing import *;

def enable_win_unicode_console():
    try:
        # Fix UTF8 output issues on Windows console.
        # Does nothing if package is not installed
        from win_unicode_console import enable
        enable()
    except ImportError:
        pass

def plot_dataframe(dataframe, column_name):
    """ Plots a dataframe's column by name (column_name) 
        with numeric values
    """
    dataframe[column_name].plot()
    plt.show()

def get_ratings_by_movie_id(ratings_dataframe, movie_id):
    """ Gets the ratings for a movie by id (movie_id) """ 
    user_ratings = ratings_dataframe[ratings_dataframe["movieId"] == movie_id]['rating']
    return user_ratings

def get_actors_data_by_movie_id(credits_dataframe, movie_id):
    """ Gets the actors data for a movie by id (movie_id) """
    raw_cast_data = credits_dataframe[credits_dataframe["id"] == int(movie_id)]["cast"]
    cast_data = ast.literal_eval(raw_cast_data.iloc[0])
    actors_data = {actor["id"]: actor["name"] for actor in cast_data}
    #print(cast_data)
    if not cast_data:
        director_data = {0:""}
    return actors_data

def get_directors_data_by_movie_id(credits_dataframe, movie_id):
    """ Gets the directors data for a movie by id (movie_id) """
    raw_crew_data = credits_dataframe[credits_dataframe["id"] == int(movie_id)]["crew"]
    crew_data = ast.literal_eval(raw_crew_data.iloc[0])
    director_data = {crew["id"]: crew["name"] for crew in crew_data if crew["job"] == "Director"}
    #print(crew_data)
    if not crew_data:
        director_data = {0:""}
    return director_data

def read_data(file_path):
    """ Reads the dara from a file path = file_path and returns a dataset """
    dataframe = pd.read_csv(file_path, low_memory=False)
    return dataframe

def get_categorical_data_encoder(data):
    """ Gets the encoded binary data into categorical """
    le = LabelEncoder()
    le.fit(data)
    #fitted_tittle = le.transform(movies_metadata_dataframe["title"][0:4])
    #list(le.inverse_transform([2, 2, 1]))
    return le

def get_all_actors_data(credits_dataframe, movies_metadata_dataframe):
    """ Gets the whole actors' data from the dataframes """
    movies_metadata_dataframe_ids = movies_metadata_dataframe["id"].values
    all_actors_data = list()
    for id in movies_metadata_dataframe_ids:
        result = get_actors_data_by_movie_id(credits_dataframe, id)
        all_actors_data.append(result.values())
    return all_actors_data

def get_all_directors_data(credits_dataframe, movies_metadata_dataframe):
    """ Gets the whole directors' data from the dataframes """
    movies_metadata_dataframe_ids = movies_metadata_dataframe["id"].values
    all_directors_data = list()
    for id in movies_metadata_dataframe_ids:
        result = get_directors_data_by_movie_id(credits_dataframe, id)
        all_directors_data.append(list(result.keys())[0]) 
    return all_directors_data

def get_one_hot_multilabled_dataframe(data_values, column_name):
    """ Parses the data_values into a one-hot multilabeled 
        dataframe that can be used in the algorthms
    """
    #df = pd.DataFrame(columns=[column_name])
    #for item in data_values:
    #    df = df.append({column_name :item}, ignore_index=True )
    
    mlb = MultiLabelBinarizer()
    mlb_result = mlb.fit_transform(data_values[column_name])
    #df1 = pd.DataFrame(mlb_result, columns=mlb.classes_, index=data_values.index)
    df1 = pd.DataFrame(mlb_result, columns=mlb.classes_)
    #print("One-hot mulilabled dataframe of columnn {0}:".format(column_name))
    #print(df1)
    #print("\n")
    return df1

def show_columns_with_nan(dataframe):
    """ Prints the columns names of the dataframe which have NaN values"""
    print("  Is any column with NaN: ")
    columns_with_nan = dataframe.columns[dataframe.isnull().any()]
    if columns_with_nan != []:
        print(dataframe.columns[dataframe.isnull().any()])
        print("\n")
    else:
        print("    No")

def create_testing_dataframe(movies_metadata_dataframe, credits_dataframe):
    """ Creates a dataframe for testing """
    print("\nCreating a testing dataframe for the algorithms ...")
    df = pd.DataFrame()
    df["production_companies"] = movies_metadata_dataframe["production_companies"].apply(lambda x: len(x))
    df["production_countries"] = movies_metadata_dataframe["production_countries"].apply(lambda x: len(x))
    df["genres"] = movies_metadata_dataframe["genres"].apply(lambda x: len(x))
    df["belongs_to_collection"] = movies_metadata_dataframe["belongs_to_collection"].apply(lambda x: len(x))
    df["runtime"] = movies_metadata_dataframe["runtime"]
    #df["runtime"] = df['runtime'].fillna(df['runtime'].mean())
    df["popularity"] = movies_metadata_dataframe["popularity"]
    df["is_english"] = movies_metadata_dataframe["is_english"]
    # Released on friday - 4 == Friday
    df["is_released_on_friday"] = movies_metadata_dataframe["day_of_week"].apply(lambda x: 1 if x == 4 else 0)
    # Released in summer - 6 == June, 7 == July, 8 == August
    df['is_released_in_summer'] = movies_metadata_dataframe['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    # Released on hoiday - 4 == April, 5 === May, 6 == June, 11 == November
    df['is_released_on_holiday'] = movies_metadata_dataframe['month'].apply(lambda x: 1 if x in [4, 5, 6, 11] else 0)
    df['vote_average'] = movies_metadata_dataframe['vote_average']
    #df['vote_average'].replace(np.nan, 0.0, inplace=True)
    #df['vote_average'] = df['vote_average'].fillna(df['vote_average'].mean())
    df['budget'] = movies_metadata_dataframe['budget']
    df['vote_count'] = movies_metadata_dataframe['vote_count']
    df['year'] = movies_metadata_dataframe['year']

    return_data = movies_metadata_dataframe["revenue"].replace(0.0, np.nan) / movies_metadata_dataframe['budget'].replace(0.0, np.nan)
    df['is_successfull'] = return_data.apply(lambda x: 1 if x >=1 else 0)
    #print(df[df['is_successfull'].isnull()].shape)
    
    # Get the actors count - VERY SLOW !!!
    #print("  Getting the actors count data ...")
    #raw_cast_data = credits_dataframe["cast"]
    #actors_count_data = [len(ast.literal_eval(item)) for item in raw_cast_data.values]
    #df["actors_count"] = pd.Series(actors_count_data)

    # Get the crew count - VERY SLOW !!!
    #print("  Getting the crew count data ...")
    #raw_crew_data = credits_dataframe["crew"]
    #crew_count_data = [len(ast.literal_eval(item)) for item in raw_crew_data.values]
    #df["crew_count"] = pd.Series(crew_count_data)
    
    # Adds the genres as a separate column
    genres_one_hot = get_one_hot_multilabled_dataframe(movies_metadata_dataframe, "genres")
    df = df.join(genres_one_hot)

    # Filters the movies based on vote_count colum and a percentile limit - NOT VERY USEFUL
    #percentile = 0.25
    #df = remove_movies_with_less_votes(df, percentile)

    # Filtering the budget using only the values 
    # greater than some percetile of the data - NOT VERY USEFUL
    #percentile = 0.50
    #print(df.shape)
    #q_budget = df['budget'].quantile(percentile)
    #df = df[df['budget'] > q_budget]
    #print(df.shape)

    columns_to_filter = ["popularity", "runtime", "vote_average",
                         "budget", "vote_count"]
    print("Filtering the dataframe using only the data that has no NaN values " +\
       "for the columns:\n {}".format(columns_to_filter))
    print("  Shape before filtering: ", df.shape)
    for item in columns_to_filter:
        print(item, " ", df[item].isnull().sum())
        df = df[df[item].notnull()]
    #df = df[(df["is_successfull"].notnull()) & (df["popularity"].notnull()) & (df["runtime"].notnull())
    #        & (df["vote_average"].notnull()) & (df["budget"].notnull())]
    print("  Shape after filtering: ", df.shape)
    
    # Prints all columns and True/False whether it constains NaN values
    #print("Columns with/without NaN: ")
    #print(df.isnull().any())
    #print("\n")

    show_columns_with_nan(df)

    print("  The columns of the dataframe are: ")
    print(df.columns)

    return df

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
    df = create_testing_dataframe(movies_metadata_dataframe, credits_dataframe)

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
    X = df.drop(columns="is_successfull")
    y = df["is_successfull"]

    test_decision_tree_classification_with_cv(X, y)
    test_gradient_boosting_classification_with_cv(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=0.33,
                                                       random_state=42)

    test_gradient_boosting_classification(X_train, X_test, y_train, y_test)

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

def remove_movies_with_less_votes(dataframe, percentile):
    """ Removes the movies with less vote count than the percentile """
    print("Filtering the movies with less vote count than the percentile = {} ...".format(percentile))
    #all_votes_count = movies_metadata_dataframe["vote_count"].values.astype('int')
    all_votes_count = dataframe[dataframe["vote_count"].notnull()]['vote_count'].astype('int')
    #all_votes_count_edited = [int(item) for item in all_votes_count]
    #average_votes_count = sum(all_votes_count_edited)/float(len(all_votes_count_edited))
    #all_votes_count_filtered_ids = [index for index, item in enumerate(all_votes_count_edited)
    #                               if item > average_votes_count]
    votes_count_limit = all_votes_count.quantile(percentile)
    print("  The {0} percentile of all vote counts: {1}".format(percentile, votes_count_limit))
    #print(votes_count_limit)
    print("  The movies count before filtering: {0}".format(len(dataframe)))
    dataframe = dataframe[dataframe["vote_count"] > votes_count_limit]
    print("  The movies after filtering: {0}".format(len(dataframe)))
    return dataframe

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
    #print("  Reading the imdb movies' credits ...")
    #imdb_movies_dataframe = read_data(imdb_movies_file_path)

    # Gets data from the imdb_movies_movies_dataframe
    #test_imdb = imdb_movies_dataframe[["tid","nrOfWins", "nrOfNominations"]]
    #repaired_test_imdb = []
    #for index, item in enumerate(imdb_movies_dataframe["nrOfWins"]):
    #    try:
    #        if item.isdigit():
    #            repaired_test_imdb.append(item)
    #        else:
    #            other = imdb_movies_dataframe.at[index,"nrOfNominations"]
    #            repaired_test_imdb.append(other)
    #    except :
    #        repaired_test_imdb.append(np.nan)
            
    #test_imdb["nrOfWins"] = pd.Series(repaired_test_imdb)
    #movies_metadata_dataframe = movies_metadata_dataframe_old.join(test_imdb.drop(columns="nrOfNominations"))

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

    # Plots a dataframe
    #plot_dataframe(movies_metadata_dataframe, "vote_count")
    #ratings_dataframe[:100]["rating"].plot(kind='bar')    
    #plt.show()

if __name__ == "__main__":
    # Enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
