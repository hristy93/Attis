import ast
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import statistics as s
from dateutil.parser import parse
import math
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
    print("accuracy_score: ", accuracy_score(y_test, y_pred))
    print("classification_report:\n", classification_report(y_test, y_pred))
    print(result)

def test_decision_tree_regression(X_train, X_test, y_train, y_test):
    clf_entropy = DecisionTreeRegressor(max_depth=2)
    clf_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    result = math.sqrt(mean_squared_error(y_test, y_pred))
    print(result)

def test_gradient_boosting_regression(X_train, X_test, y_train, y_test):
    reg = GradientBoostingRegressor()
    reg.fit(X_train, y_train)
    result = reg.score(X_test, y_test)
    print(result)

def get_cross_validation_score(classificator, X, y):
    k_fold_count = 10
    scores = cross_val_score(classificator, X, y, cv=k_fold_count)
    print("{0}-fold cross validation scores: {1}".format(k_fold_count, scores))
    print("average score: {0}".format(s.mean(scores)))

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

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

def edit_data_values(data_values, value_type = "int"):
    data_values_edited = []
    for item in data_values:
        #if item == '0' or item == 0 or\
        #   item == 0.0 or math.isnan(item):
        #   item = np.nan
        if isinstance(item, str):
            if value_type == "float":
                if item.replace('.','', 1).isdigit():
                    item = float(item)
                else:
                    item = np.nan
            elif value_type == "int":
                if item.isdigit():
                    item = int(item)
                else:
                    item = np.nan
        #else:
        #    #item = sum(float_data_edited)/float(len(float_data_edited)
        #    item = np.nan

        data_values_edited.append(item)

    #if value_type == "int":
    #    data_values_series = pd.Series(data_values_edited, dtype=np.int32)
    #else:
    #    data_values_series = pd.Series(data_values_edited, dtype=np.float)

    #print(data_values_series.head(10))
    #data_values_edited_series = data_values_series[data_values_series != np.nan].astype(value_type)
    #return data_values_edited_series

    data_values_series = pd.Series(data_values_edited)
    return data_values_series



def test_decision_trees(credits_dataframe, movies_metadata_dataframe):
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
    df = pd.DataFrame()
    df["production_companies"] = movies_metadata_dataframe["production_companies"].apply(lambda x: len(x))
    df["production_countries"] = movies_metadata_dataframe["production_countries"].apply(lambda x: len(x))
    df["genres"] = movies_metadata_dataframe["genres"].apply(lambda x: len(x))
    df["belongs_to_collection"] = movies_metadata_dataframe["belongs_to_collection"].apply(lambda x: len(x))
    df["runtime"] = movies_metadata_dataframe["runtime"]
    df["popularity"] = movies_metadata_dataframe["popularity"]
    df["is_english"] = movies_metadata_dataframe["is_english"]
    df["is_released_on_friday"] = movies_metadata_dataframe["day_of_week"].apply(lambda x: 1 if x == 4 else 0)
    df['is_released_in_summer'] = movies_metadata_dataframe['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)

    # Get the actors count - VERY SLOW !!!
    #raw_cast_data = credits_dataframe["cast"]
    #actors_count_data = [len(ast.literal_eval(item)) for item in raw_cast_data.values]
    #df["actors_count"] = pd.Series(actors_count_data)
 
    genres_one_hot = get_one_hot_multilabled_dataframe(movies_metadata_dataframe, "genres")
    df = df.join(genres_one_hot)
    print(df.isnull().any())
  

    # Test 2.1
    #average = 68787389
    #df = df[df["revenue"] != average]
    #X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="revenue"),
    #                                                   df["revenue"],
    #                                                   test_size=0.33,
    #                                                   random_state=42)
    #test_decision_tree_regression(X_train, X_test, y_train, y_test)
    #test_gradient_boosting_regression(X_train, X_test, y_train, y_test)


    # Test 2.2
    return_data = movies_metadata_dataframe["revenue"].replace(0.0, np.nan) / movies_metadata_dataframe['budget'].replace(0.0, np.nan)
    df["return_ration"] = return_data
    print(df[df['return_ration'].isnull()].shape)
    #rio_data = (movies_metadata_dataframe["revenue"].replace(0.0, np.nan) - movies_metadata_dataframe['budget'].replace(0.0, np.nan) ) / movies_metadata_dataframe['budget'].replace(0.0, np.nan)
    #df["return_ration"] = rio_data
    df = df[(df['return_ration'].notnull()) & (df["popularity"].notnull()) & (df["runtime"].notnull())]
    df['return_ration'] = df['return_ration'].apply(lambda x: 1 if x >=1 else 0)
    print(df.shape)
    print(df.isnull().any())

    k_fold_count = 10
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
    max_depth=3, min_samples_leaf=5)
    get_cross_validation_score(clf_entropy, df.drop(columns="return_ration"), df["return_ration"])

     # Test 2.3
    #movies_metadata_dataframe["revenue"] = movies_metadata_dataframe["revenue"].replace(0.0, np.nan)
    #print(df[df['revenue'].isnull()].shape)
    #df = df[(df['revenue'].notnull()) & (df["popularity"].notnull()) & (df["runtime"].notnull())]
    #print(df.shape)
    #print(df.isnull().any())
    #X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="revenue"),
    #                                                   df["revenue"],
    #                                                   test_size=0.33,
    #                                                   random_state=42)
    #reg = GradientBoostingRegressor()
    #get_cross_validation_score(reg, df.drop(columns="revenue"), df["revenue"])
    #test_decision_tree_regression(X_train, X_test, y_train, y_test)
    #test_gradient_boosting_regression(X_train, X_test, y_train, y_test)
    

    # Test 3
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
    
    #test_decision_tree(X_train, X_test, y_train, y_test)
    #test_decision_tree_regression(X_train, X_test, y_train, y_test)
    #test_gradient_boosting_regression(X_train, X_test, y_train, y_test)

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

def parse_data_in_column(movies_metadata_dataframe, column_name, item_name):
    data_values = movies_metadata_dataframe[column_name].fillna("[]").values
    data_values_parsed = []
    for item in data_values:
        try:
            data_values_evaluated = ast.literal_eval(item)
            if isinstance(data_values_evaluated, list):
                result = [item[item_name] for item in data_values_evaluated ]
            elif isinstance(data_values_evaluated, dict):
                result = [data_values_evaluated[item_name]]
            data_values_parsed.append(result)
        except:
            data_values_evaluated = []
            data_values_parsed.append([])
    #print(data_values_parsed)
    return pd.Series(data_values_parsed)

def preprocess_dataset_column(dataframe, column_name, is_float, fill_na):
    # print("  Prepocessing the {0} data ...".format(column_name))
    if is_float:
        column_data = edit_data_values(dataframe[column_name], "float")
    else:
        column_data = edit_data_values(dataframe[column_name])

    if fill_na:
        column_data_mean = column_data[column_data != np.nan].mean()
        column_data = column_data.fillna(column_data_mean)
        #if is_float:
        #    column_data = column_data.fillna(column_data_mean)
        #else:
        #    column_data = column_data.fillna(int(column_data_mean))

    dataframe[column_name] = column_data

def preprocess_movies_metadata(movies_metadata_dataframe, fill_na = False):
    print("\nPreprocessing movies' metadata ...")

    # Print the shape of the dataframe
    print("  Movies metadata dataframe shape before preprocessing: {0}".format(movies_metadata_dataframe.shape))

    # Parsing release_date data
    print("  Prepocessing the release_date data ...")
    release_date_data = movies_metadata_dataframe["release_date"]
    day_of_week_data = []
    month_data = []
    for date in release_date_data:
        if isinstance(date, str):
            #day_of_week_number = datetime.datetime.strptime(date, "%m/%d/%Y").weekday()
            parsed_date = parse(date)
            day_of_week_number = parsed_date.weekday()
            month_number = parsed_date.month
            day_of_week_data.append(day_of_week_number)
            month_data.append(month_data)
        elif np.isnan(date):
            day_of_week_data.append(-1)
            month_data.append(-1)

    movies_metadata_dataframe["day_of_week"] = pd.Series(day_of_week_data)
    movies_metadata_dataframe["month"] = pd.Series(month_data)



    #is_friday_data = [datetime.datetime.strptime(date, "%m/%d/%Y").weekday() if date != np.nan else -1 for date in release_date_data ]

    # Removing useless columns
    columns_to_remove = ["homepage", "imdb_id", "original_title", "overview", "poster_path",
                        "tagline", "video"]
    print("  Removing useless columns : {0}".format(columns_to_remove))
    movies_metadata_dataframe = movies_metadata_dataframe.drop(columns_to_remove, axis=1)

    # Removing the adult column
    adult_movies_count = len([item for item in movies_metadata_dataframe["adult"] if item != "FALSE"])
    print("  Removing the 'adult' column - there are just {} adult movies".format(adult_movies_count))
    movies_metadata_dataframe = movies_metadata_dataframe.drop("adult", axis=1)

    # Parsing production_companies data
    print("  Preprocessing the production_companies data ...")
    movies_metadata_dataframe["production_companies"] = parse_data_in_column(movies_metadata_dataframe, "production_companies", "name")

    # Parsing production_countries data
    print("  Preprocessing the production_countries data ...")
    movies_metadata_dataframe["production_countries"] = parse_data_in_column(movies_metadata_dataframe, "production_countries", "name")

    # Parsing genres data
    print("  Preprocessing the genres data ...")
    movies_metadata_dataframe["genres"] = parse_data_in_column(movies_metadata_dataframe, "genres", "name")

    # Parsing belongs_to_collection data
    print("  Prepocessing the belongs_to_collection data ...")
    movies_metadata_dataframe["belongs_to_collection"] = parse_data_in_column(movies_metadata_dataframe, "belongs_to_collection", "name")

    #one_hot_multilabled_genres_dataframe =\
    #    get_one_hot_multilabled_dataframe(movies_metadata_dataframe["genres"], "genres")

    # Preprocessing the original_language data
    print("  Preprocessing the original_language data ...")
    original_language_data = movies_metadata_dataframe["original_language"]
    original_language_data_values = original_language_data.values
    is_english_data = [0 if item != "en" else 1 for item in original_language_data_values]
    is_english_data_series = pd.Series(is_english_data);
    movies_metadata_dataframe["is_english"] = is_english_data_series
    #print(movies_metadata_dataframe["is_english"].describe())

    # Preprocessing the vote_average data
    print("  Preprocessing the vote_average data ...")
    #vote_average_data = movies_metadata_dataframe["vote_average"].astype("float")
    preprocess_dataset_column(movies_metadata_dataframe, "vote_average", True, fill_na)
    #print(movies_metadata_dataframe["vote_average"].describe())

    # Preprocessing the runtime data
    print("  Preprocessing the revenue data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "runtime", False, fill_na)
    #print(runtime_data.isnull().any())
    #print(runtime_data.isnull())
    #print(movies_metadata_dataframe["runtime"].describe())

    # Preprocessing the popularity data
    print("  Prepocessing the popularity data ...")
    #popularity_data = movies_metadata_dataframe["popularity"].astype("float")
    preprocess_dataset_column(movies_metadata_dataframe, "popularity", True, fill_na)
    #print(movies_metadata_dataframe["popularity"].describe())

    # Preprocessing the revenue data
    print("  Preprocessing the revenue data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "revenue", False, fill_na)
    #print(movies_metadata_dataframe["revenue"].describe())

    # Preprocessing the budget data
    print("  Preprocessing the budget data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "budget", False, fill_na)
    #print(movies_metadata_dataframe["budget"].describe())

    # Print the dataframe
    #print("Print movies' metadata dataframe:")
    #print(movies_metadata_dataframe)

    # Print the shape of the dataframe
    print("  Movies metadata dataframe shape after preprocessing: {0}".format(movies_metadata_dataframe.shape))

    print("  Are there any NAN values in the movie metadata : " +
          str(movies_metadata_dataframe.isnull().values.any()))

    return movies_metadata_dataframe

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

    return credits_dataframe

def main():
    movies_metadata_test_file_path = "movies_metadata_test.csv"
    movies_metadata_file_path = "files/the-movies-dataset/movies_metadata.csv"
    credits_file_path = "files/the-movies-dataset/credits.csv"
    ratings_file_path = "files/the-movies-dataset/ratings_small.csv"

    print("Reading movies data ...")

    # Read the movies metadata data
    print("  Reading the movies' metadata ...")
    movies_metadata_dataframe = read_data(movies_metadata_file_path)

    # Read the ratings data
    print("  Reading the movies' ratings ...")
    ratings_dataframe = read_data(ratings_file_path)

    # Read the credits data
    print("  Reading the movies' credits ...")
    credits_dataframe = read_data(credits_file_path)

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
    test_decision_trees(credits_dataframe, movies_metadata_dataframe)

    # Plots a dataframe
    #plot_dataframe(movies_metadata_dataframe, "vote_count")
    #ratings_dataframe[:100]["rating"].plot(kind='bar')    
    #plt.show()

if __name__ == "__main__":
    # Enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
