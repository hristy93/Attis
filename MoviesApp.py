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
import statistics as s
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
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

def test_decision_tree(X_train, X_test, y_train, y_test):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
    max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    result = accuracy_score(y_test, y_pred)
    print(result)

def test_decision_tree_regression(X_train, X_test, y_train, y_test):
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

def get_one_hot_multilabled_dataframe(data_values, column_name):
    df = pd.DataFrame(columns=[column_name])
    for item in data_values:
        df = df.append({column_name :item}, ignore_index=True )
    
    mlb = MultiLabelBinarizer()
    mlb_result = mlb.fit_transform(df[column_name])
    df1 = pd.DataFrame(mlb_result, columns=mlb.classes_, index=df.index)
    print("One-hot mulilabled dataframe:")
    print(df1)
    print("\n")
    return df1

def edit_data_values(data_values):
    data_values_edited = []
    for item in data_values:
        if item.replace('.','', 1).isdigit():
            if item == '0':
                item = np.nan
        else:
            #item = sum(float_data_edited)/float(len(float_data_edited)
            item = np.nan
        data_values_edited.append(float(item))
    return pd.Series(data_values_edited)

def test_decision_tree(credits_dataframe, movies_metadata_dataframe):
    all_popularity_data = movies_metadata_dataframe["popularity"].values
    #all_popularity_data.replace("he uses this to woo local beauty Beatrice.", 10.168437)
    #np.place(all_popularity_data, [str.isdecimal(item) for item in all_popularity_data], 10.168437)
    all_popularity_data_edited = edit_data_values(all_popularity_data)
    #all_popularity_data[57] = '10.168437'
    
    df1 = pd.DataFrame({"popularity": all_popularity_data_edited})
    vote_average = edit_data_values(movies_metadata_dataframe["vote_average"].values)
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
    
    test_decision_tree_regression(X_train, X_test, y_train, y_test)

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

def association_rules_test(dataframe, support):
    print("\nTesting asocciation rules ...")
    frequent_itemsets = apriori(dataframe, min_support=support, use_colnames=True)
    print("  Frequent itemsets:")
    print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print("  Frequent itemsets:")
    print(rules.head())

def remove_movies_with_less_votes(movies_metadata_dataframe, quantile):
    #all_votes_count = movies_metadata_dataframe["vote_count"].values.astype('int')
    all_votes_count = movies_metadata_dataframe[movies_metadata_dataframe["vote_count"].notnull()]['vote_count'].astype('int')
    #all_votes_count_edited = [int(item) for item in all_votes_count]
    #average_votes_count = sum(all_votes_count_edited)/float(len(all_votes_count_edited))
    #all_votes_count_filtered_ids = [index for index, item in enumerate(all_votes_count_edited)
    #                               if item > average_votes_count]
    votes_count_limit = all_votes_count.quantile(quantile)
    print("The {0} percentile is:".format(votes_count_limit))
    print(votes_count_limit)
    movies_metadata_dataframe = movies_metadata_dataframe[movies_metadata_dataframe["vote_count"] > votes_count_limit]
    remaining_movies_count = len(movies_metadata_dataframe)
    print("The remaining movies are {0}".format(remaining_movies_count))
    return movies_metadata_dataframe

def parse_list_data_in_column(movies_metadata_dataframe, column_name, item_name):
    data_values = movies_metadata_dataframe[column_name].fillna("[]").values
    data_values_parsed = []
    for item in data_values:
        try:
            data_values_evaluated = ast.literal_eval(item)
            result = [item[item_name] for item in data_values_evaluated ]
            data_values_parsed.append(result)
        except:
            data_values_evaluated = []
            data_values_parsed.append([])
    #print(data_values_parsed)
    return pd.Series(data_values_parsed)

def preprocess_movies_metadata(movies_metadata_dataframe):
    print("\nPreprocessing movies' metadata ...")

    # Print the shape of the dataframe
    print("  Credits dataframe shape before preprocessing:")
    print(movies_metadata_dataframe.shape)

    # Removing useless columns
    columns_to_remove = ["homepage", "imdb_id", "original_title", "overview", "poster_path",
                        "tagline", "video"]
    movies_metadata_dataframe = movies_metadata_dataframe.drop(columns_to_remove, axis=1)

    # Removing the adult column
    adult_movies_count = len([item for item in movies_metadata_dataframe["adult"] if item != "FALSE"])
    print("  Removing the 'adult' column - there are just {} adult movies".format(adult_movies_count))
    movies_metadata_dataframe = movies_metadata_dataframe.drop("adult", axis=1)

    # Parsing production companies data
    movies_metadata_dataframe["production_companies"] = parse_list_data_in_column(movies_metadata_dataframe, "production_companies", "name")

    # Parsing production countries data
    movies_metadata_dataframe["production_countries"] = parse_list_data_in_column(movies_metadata_dataframe, "production_countries", "name")

    # Parsing genres countries data
    movies_metadata_dataframe["genres"] = parse_list_data_in_column(movies_metadata_dataframe, "genres", "name")

    #one_hot_multilabled_genres_dataframe =\
    #    get_one_hot_multilabled_dataframe(movies_metadata_dataframe["genres"], "genres")

    # Replace incorrect vote average values with the mean of the vote average values
    vote_average_data = edit_data_values(movies_metadata_dataframe["vote_average"])
    vote_average_data = vote_average_data.fillna(s.mean(vote_average_data))
    movies_metadata_dataframe["vote_average"] = vote_average_data
    print(movies_metadata_dataframe["vote_average"].describe())

    # Replace incorrect runtime values with the mean of the runtime values
    runtime_data = movies_metadata_dataframe["runtime"].fillna(movies_metadata_dataframe["runtime"].mean())
    print(movies_metadata_dataframe["runtime"].describe())
 
    # Print the dataframe
    #print("Print movies' metadata dataframe:")
    #print(movies_metadata_dataframe)

    # Print the shape of the dataframe
    print("  Credits dataframe shape after preprocessing:")
    print("  {0}".format(movies_metadata_dataframe.shape))

def preprocess_movies_credits(credits_dataframe):
    print("\nPreprocessing credits' metadata ...")

    # Make integer columns as int type
    credits_dataframe["id"].astype("int")

    # Print the movies' credits dataframe
    #print("Print credits dataframe:")
    #print(credits_dataframe)

    # Print the shape of the dataframe
    print("  Credits dataframe shape:")
    print("  {0}".format(credits_dataframe.shape))

def main():
    movies_metadata_test_file_path = "movies_metadata_test.csv"
    movies_metadata_file_path = "../../movies/the-movies-dataset/movies_metadata.csv"
    credits_file_path = "../../movies/the-movies-dataset/credits.csv"
    ratings_file_path = "../../movies/the-movies-dataset/ratings_small.csv"

    print("Reading movies data ...")

    # Read the movies metadata data
    print("  Reading the movies' metadata ...")
    movies_metadata_dataframe = read_data(movies_metadata_test_file_path)

    # Read the ratings data
    print("  Reading the movies' ratings ...")
    ratings_dataframe = read_data(ratings_file_path)

    # Read the credits data
    print("  Reading the movies' credits ...")
    credits_dataframe = read_data(credits_file_path)

    # Preprocesses the movies' metadata
    preprocess_movies_metadata(movies_metadata_dataframe)

    # Preprocesses the movies' credits
    preprocess_movies_credits(credits_dataframe)

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
    #test_decision_tree(credits_dataframe, movies_metadata_dataframe)

    # Plots a dataframe
    #plot_dataframe(movies_metadata_dataframe, "vote_count")
    #ratings_dataframe[:100]["rating"].plot(kind='bar')    
    #plt.show()

if __name__ == "__main__":
    # Enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
