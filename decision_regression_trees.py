import numpy as np
import pandas as pd

from utils import *

def create_trees_testing_dataframe(movies_metadata_dataframe, credits_dataframe):
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
    #df["is_english"] = movies_metadata_dataframe["is_english"]
    # Released on friday - 4 == Friday
    #df["is_released_on_friday"] = movies_metadata_dataframe["day_of_week"].apply(lambda x: 1 if x == 4 else 0)
    # Released in summer - 6 == June, 7 == July, 8 == August
    #df["is_released_in_summer"] = movies_metadata_dataframe["month"].apply(lambda x: 1 if x in [6, 7, 8, 11] else 0)
    # Released on hoiday - 4 == April, 5 == May, 6 == June, 11 == November
    #df["is_released_on_holiday"] = movies_metadata_dataframe["month"].apply(lambda x: 1 if x in [4, 5, 6] else 0)
    df["vote_average"] = movies_metadata_dataframe["vote_average"]
    #df['vote_average'].replace(np.nan, 0.0, inplace=True)
    #df['vote_average'] = df['vote_average'].fillna(df['vote_average'].mean())
    df["budget"] = movies_metadata_dataframe["budget"].apply(scale_small_values)
    #df["directors"] = credits_dataframe["directors"]

    # Filters the budget by some proportion of its mean
    # mean_budget = movies_metadata_dataframe['budget'].mean()
    # print("Mean budget for all movies", mean_budget)
    # df["is_big_budget"] = movies_metadata_dataframe["budget"].apply(
    #     lambda b: b > mean_budget)
    # df["is_enourmous_budget"] = movies_metadata_dataframe["budget"].apply(
    #     lambda b: b > 1.3 * mean_budget)
    # df["is_low_budget"] = movies_metadata_dataframe["budget"].apply(
    #     lambda b: b < 0.75 * mean_budget)

    # Released in winter - 10 == October, 11 == November, 12 == December, 1 == January,
    # 2 == February, 3 == March
    #df["is_released_in_winter"] = movies_metadata_dataframe["month"].apply(
    #    lambda x: 1 if x in [10, 11, 12, 1, 2, 3] else 0)

    df["vote_count"] = movies_metadata_dataframe["vote_count"]

    # There is an improvement of 0.7% on the 
    # gradient boosting classificatior without the year
    #df["year"] = movies_metadata_dataframe["year"] 

    # New features from the imdb movies file
    #df["nrOfWins"] = movies_metadata_dataframe["nrOfWins"]
    #df["nrOfNominations"] = movies_metadata_dataframe["nrOfNominations"]
    #df["nrOfPhotos"] = movies_metadata_dataframe["nrOfPhotos"]
    #df["nrOfNewsArticles"] = movies_metadata_dataframe["nrOfNewsArticles"]
    #df["nrOfUserReviews"] = movies_metadata_dataframe["nrOfUserReviews"]

    return_data = movies_metadata_dataframe["revenue"].replace(0.0, np.nan) / \
        movies_metadata_dataframe['budget'].replace(0.0, np.nan)
    df["return"] = return_data
    #print("Revenue Nan: ", movies_metadata_dataframe[movies_metadata_dataframe["revenue"].isnull()].shape)
    #print("budget Nan: ", movies_metadata_dataframe[movies_metadata_dataframe["budget"].isnull()].shape)
    #print(return_data.describe())
    #print(df[df["return"].notnull()].shape)
    print("  Shape before filtering: ", df.shape)
    df = df[df["return"].notnull()]
    print("  Shape after filtering: ", df.shape)
    df.drop(columns=["return"], inplace=True)

    df['is_successful'] = return_data.apply(lambda x: 1 if x >= 1 else 0)
    #print(df[df['is_successful'].notnull()].shape)
    #print(df['is_successful'].describe())
    
    ## Get the actors count - VERY SLOW !!!
    #print("  Getting the actors count data ...")
    #raw_cast_data = credits_dataframe["cast"]
    #actors_count_data = [len(ast.literal_eval(item)) for item in raw_cast_data.values]
    #df["actors_count"] = pd.Series(actors_count_data)

    ## Get the crew count - VERY SLOW !!!
    #print("  Getting the crew count data ...")
    #raw_crew_data = credits_dataframe["crew"]
    #crew_count_data = [len(ast.literal_eval(item)) for item in raw_crew_data.values]
    #df["crew_count"] = pd.Series(crew_count_data)
    
    # Adds some of the genres as a separate column
    #genres_one_hot = get_one_hot_multilabled_dataframe(movies_metadata_dataframe, "genres")
    #print(genres_one_hot.columns)
    #colums_to_remove = ["Aniplex", "BROSTA TV",
    #   "Carousel Productions", "GoHands", "Sentai Filmworks", "The Cartel",
    #   "Vision View Entertainment",  "Sentai Filmworks", "Rogue State",
    #   "Mardock Scramble Production Committee", "GoHands", "Odyssey Media",
    #   "Pulser Productions", "Telescene Film Group Productions"]
    #genres_one_hot_filted = genres_one_hot.drop(columns=colums_to_remove)
    #df = df.join(genres_one_hot_filted)

    # Filters the movies based on vote_count colum and a percentile limit - LITTLE IMPROVEMENT
    #percentile = 0.20
    #df = remove_movies_with_less_votes(df, percentile)

    # Filtering the budget using only the values 
    # greater than some percetile of the data - NOT VERY USEFUL
    #percentile = 0.50
    #print(df.shape)
    #q_budget = df['budget'].quantile(percentile)
    #df = df[df['budget'] > q_budget]
    #print(df.shape)

    columns_to_filter = ["popularity", "runtime", "vote_average",
                         "budget", "vote_count", "directors"]

    # New columns from the imdb movies file
    #new_columns = ["nrOfWins", "nrOfNominations",
    #                     "nrOfPhotos", "nrOfNewsArticles", "nrOfUserReviews"]
    #columns_to_filter.extend(new_columns)

    print("Filtering the dataframe using only the data that has no NaN values " +\
       "for the columns:\n {}".format(columns_to_filter))
    print("  Shape before filtering: ", df.shape)
    print("  The NaN values in each column are: ")
    for item in columns_to_filter:
        if item in df.columns:
            print(item, " ", df[item].isnull().sum())
            df = df[df[item].notnull()]
    print("  Shape after filtering: ", df.shape)
    
    # Prints all columns and True/False whether it constains NaN values
    #print("Columns with/without NaN: ")
    #print(df.isnull().any())
    #print("\n")

    show_columns_with_nan(df)

    print("  The columns of the dataframe are: ")
    print(df.columns)

    return df
