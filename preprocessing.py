import ast
import numpy as np
import pandas as pd
from dateutil.parser import parse

def edit_data_values(data_values, value_type = "int"):
    """ Edits the values of the data (data_values) depending
        on whether their type (value_type) is int or float 
    """
    data_values_edited = []
    for item in data_values:
        # Makes all data with the value of 0 to be NaN
        if item == '0' or item == 0 or item == 0.0:
           item = np.nan
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

def parse_data_in_column(movies_metadata_dataframe, column_name, item_name):
    """ Parses the data in the column column_name which contains a list/dict
        of value and gets the item_name's value(s)
    """
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
    """ Preprocesses the column name column_name in the dataframe.
        It fills the invalid data with the mean of the data if full_na = true.
        Calls the appropriate function when checking if the data is float
    """
    # print("  Preprocessing the {0} data ...".format(column_name))
    if is_float:
        column_data = edit_data_values(dataframe[column_name], "float")
    else:
        column_data = edit_data_values(dataframe[column_name])

    if fill_na:
        # Fills the NaN values with the mean of all values in the column
        column_data_mean = column_data[column_data != np.nan].mean()
        column_data = column_data.fillna(column_data_mean)
        #if is_float:
        #    column_data = column_data.fillna(column_data_mean)
        #else:
        #    column_data = column_data.fillna(int(column_data_mean))

    dataframe[column_name] = column_data

def preprocess_movies_metadata(movies_metadata_dataframe, fill_na = False):
    """ Preprocesses the movies metadata """
    print("\nPreprocessing movies' metadata ...")

    # Print the shape of the dataframe
    print("  Movies metadata dataframe shape before preprocessing: {0}".format(movies_metadata_dataframe.shape))

    # Removing useless columns
    columns_to_remove = ["homepage", "imdb_id", "original_title", "overview", "poster_path",
                         "tagline", "video", "adult"]
    print("  Removing useless columns : {0}".format(columns_to_remove))
    movies_metadata_dataframe = movies_metadata_dataframe.drop(columns_to_remove, axis=1)

    # Parsing release_date data
    print("  Preprocessing the release_date data ...")
    release_date_data = movies_metadata_dataframe["release_date"]
    day_of_week_data = []
    month_data = []
    year_data = []
    for date in release_date_data:
        if isinstance(date, str):
            #day_of_week_number = datetime.datetime.strptime(date, "%m/%d/%Y").weekday()
            parsed_date = parse(date)
            day_of_week_number = parsed_date.weekday()
            month_number = parsed_date.month
            day_of_week_data.append(day_of_week_number)
            month_data.append(month_data)
            year = parsed_date.year;
            year_data.append(year)
        elif np.isnan(date):
            day_of_week_data.append(-1)
            month_data.append(-1)
            year_data.append(-1)

    movies_metadata_dataframe["day_of_week"] = pd.Series(day_of_week_data)
    movies_metadata_dataframe["month"] = pd.Series(month_data)
    movies_metadata_dataframe["year"] = pd.Series(year_data)

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
    print("  Preprocessing the belongs_to_collection data ...")
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

    # Preprocessing the vote_count data
    print("  Preprocessing the vote_count data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "vote_count", False, fill_na)
    #print(vote_count_data.isnull().any())
    #print(vote_count_data.isnull())
    #print(movies_metadata_dataframe["vote_count"].describe())

    # Preprocessing the runtime data
    print("  Preprocessing the runtime data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "runtime", False, fill_na)
    #print(runtime_data.isnull().any())
    #print(runtime_data.isnull())
    #print(movies_metadata_dataframe["runtime"].describe())

    # Preprocessing the popularity data
    print("  Preprocessing the popularity data ...")
    #popularity_data = movies_metadata_dataframe["popularity"].astype("float")
    preprocess_dataset_column(movies_metadata_dataframe, "popularity", True, fill_na)
    #print(movies_metadata_dataframe["popularity"].describe())

    # Preprocessing the revenue data
    print("  Preprocessing the revenue data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "revenue", False, fill_na)
    #print(movies_metadata_dataframe["revenue"].describe())
    #movies_small_revenue = movies_metadata_dataframe[movies_metadata_dataframe["revenue"] <= 1]
    #print(movies_small_revenue["revenue"].count())
    #print(movies_small_revenue[["release_date", "title", "revenue"]].head(20))

    # Preprocessing the budget data
    print("  Preprocessing the budget data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "budget", False, fill_na)
    #print(movies_metadata_dataframe["budget"].describe())
    #movies_small_budget = movies_metadata_dataframe[movies_metadata_dataframe["revenue"] <= 1]
    #print(movies_small_budget["budget"].count())
    #print(movies_small_budget[["release_date", "title", "budget"]].head(20))

    # Print the dataframe
    #print("Print movies' metadata dataframe:")
    #print(movies_metadata_dataframe)

    # Print the shape of the dataframe
    print("  Movies metadata dataframe shape after preprocessing: {0}".format(movies_metadata_dataframe.shape))

    print("  Are there any NAN values in the movie metadata : " +
          str(movies_metadata_dataframe.isnull().values.any()))

    return movies_metadata_dataframe

def preprocess_movies_credits(credits_dataframe):
    """ Preprocesses the movies credits dataframe """
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
