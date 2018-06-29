import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as s
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


def enable_win_unicode_console():
    try:
        # Fix UTF8 output issues on Windows console.
        # Does nothing if package is not installed
        from win_unicode_console import enable
        enable()
    except ImportError:
        pass


def show_cross_validation_score(classificator, X, y):
    """ Shows the 10-fold cross validation scores and their
        average using the classificator and some partial (X)
        and target (y) values
    """
    k_fold_count = 10
    scores = cross_val_score(classificator, X, y, cv=k_fold_count, n_jobs=-1)
    print("  {0}-fold cross validation scores: {1}".format(k_fold_count, scores))
    average_score = s.mean(scores)
    print("  Average score: {0}".format(average_score))
    
    return average_score


def scale_small_values(value):
    """ Scales the value up """
    if value < 100:
        return value * 1000000
    elif value in range(100, 1000):
        return value * 1000
    else:
        return value


def show_columns_with_nan(dataframe):
    """ Prints the columns names of the dataframe which have NaN values"""
    print("  Is any column with NaN: ")
    columns_with_nan = dataframe.columns[dataframe.isnull().any()]
    if columns_with_nan != []:
        print(dataframe.columns[dataframe.isnull().any()])
        print("\n")
    else:
        print("    No")


def get_one_hot_multilabled_dataframe(data_values, column_name):
    """ Parses the data_values into a one-hot multilabeled
        dataframe that can be used in the algorthms
    """
    # df = pd.DataFrame(columns=[column_name])
    # for item in data_values:
    #    df = df.append({column_name :item}, ignore_index=True )

    mlb = MultiLabelBinarizer()
    mlb_result = mlb.fit_transform(data_values[column_name])
    # df1 = pd.DataFrame(mlb_result, columns=mlb.classes_, index=data_values.index)
    df1 = pd.DataFrame(mlb_result, columns=mlb.classes_)
    # print("One-hot mulilabled dataframe of columnn {0}:".format(column_name))
    # print(df1)
    # print("\n")
    return df1


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
    if not movie_id.isdigit():
        return {-1:""}
    raw_cast_data = credits_dataframe[credits_dataframe["id"] == int(movie_id)]["cast"]
    cast_data = ast.literal_eval(raw_cast_data.iloc[0])
    actors_data = {actor["id"]: actor["name"] for actor in cast_data}
    #print(cast_data)
    if not cast_data:
        director_data = {-1:""}
    return actors_data


def get_directors_data_by_movie_id(credits_dataframe, movie_id):
    """ Gets the directors data for a movie by id (movie_id) """
    if not movie_id.isdigit():
        return  {-1:""}
    try:
        raw_crew_data = credits_dataframe[credits_dataframe["id"] == int(movie_id)]["crew"]
        if raw_crew_data.empty:
            return {-1:""}
        crew_data = ast.literal_eval(raw_crew_data.iloc[0])
        director_data = {crew["id"]: crew["name"] for crew in crew_data if crew["job"] == "Director"}
        #print(crew_data)
        if not crew_data:
            return {-1:""}
    except:
       return {-1:""}
    return director_data


def read_data(file_path, encoding='utf-8'):
    """ Reads the dara from a file path = file_path and returns a dataset """
    dataframe = pd.read_csv(file_path, low_memory=False, encoding=encoding)
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
        if not result:
            all_actors_data.append(np.nan)
        else:
            all_actors_data.append(result.values())
    return all_actors_data


def get_all_directors_data(credits_dataframe, movies_metadata_dataframe):
    """ Gets the whole directors' data from the dataframes """
    movies_metadata_dataframe_ids = movies_metadata_dataframe["id"].values
    all_directors_data = list()
    for id in movies_metadata_dataframe_ids:
        result = get_directors_data_by_movie_id(credits_dataframe, id)
        if not result:
            all_directors_data.append(np.nan)
        else:
            all_directors_data.append(list(result.keys())[0])
    return all_directors_data


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
