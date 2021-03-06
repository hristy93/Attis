import ast
import numpy as np
import pandas as pd
from dateutil.parser import parse

def edit_data_values(data_values, value_type = "int", zero_to_nan=True):
    """ Edits the values of the data (data_values) depending
        on whether their type (value_type) is int or float 
    """
    data_values_edited = []
    for item in data_values:
        # Makes all data with the value of 0 to be NaN
        if zero_to_nan and item == '0' or item == 0 or item == 0.0:
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
        #    # Replaces the item with the mean of all data values
        #    item = sum(float_data_edited)/float(len(float_data_edited)

        data_values_edited.append(item)

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
    if is_float:
        column_data = edit_data_values(dataframe[column_name], "float")
    else:
        column_data = edit_data_values(dataframe[column_name])

    if fill_na:
        # Fills the NaN values with the mean of all values in the column
        column_data_mean = column_data[column_data != np.nan].mean()
        column_data = column_data.fillna(column_data_mean)

    dataframe[column_name] = column_data

def get_directors_ids(column):
    for item in column:
        if item['job'] == 'Director':
            return item['id']
        else:
            return np.nan

def preprocess_movies_metadata(movies_metadata_dataframe, fill_na = False):
    """ Preprocesses the movies metadata """
    print("\nPreprocessing movies' metadata ...")

    # Prints the shape of the dataframe
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
    movies_metadata_dataframe["production_companies"] = parse_data_in_column(movies_metadata_dataframe,
                                                                            "production_companies", "name")

    # Parsing production_countries data
    print("  Preprocessing the production_countries data ...")
    movies_metadata_dataframe["production_countries"] = parse_data_in_column(movies_metadata_dataframe,
                                                                            "production_countries", "name")

    # Parsing genres data
    print("  Preprocessing the genres data ...")
    movies_metadata_dataframe["genres"] = parse_data_in_column(movies_metadata_dataframe,
                                                              "genres", "name")

    # Parsing belongs_to_collection data
    print("  Preprocessing the belongs_to_collection data ...")
    movies_metadata_dataframe["belongs_to_collection"] = parse_data_in_column(movies_metadata_dataframe,
                                                                             "belongs_to_collection",
                                                                             "name")

    # Preprocessing the original_language data
    print("  Preprocessing the original_language data ...")
    original_language_data = movies_metadata_dataframe["original_language"]
    original_language_data_values = original_language_data.values
    is_english_data = [0 if item != "en" else 1 for item in original_language_data_values]
    is_english_data_series = pd.Series(is_english_data);
    movies_metadata_dataframe["is_english"] = is_english_data_series

    # Preprocessing the vote_average data
    print("  Preprocessing the vote_average data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "vote_average", True, fill_na)

    # Preprocessing the vote_count data
    print("  Preprocessing the vote_count data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "vote_count", False, fill_na)

    # Preprocessing the runtime data
    print("  Preprocessing the runtime data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "runtime", False, fill_na)

    # Preprocessing the popularity data
    print("  Preprocessing the popularity data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "popularity", True, fill_na)

    # Preprocessing the revenue data
    print("  Preprocessing the revenue data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "revenue", False, fill_na)

    # Preprocessing the budget data
    print("  Preprocessing the budget data ...")
    preprocess_dataset_column(movies_metadata_dataframe, "budget", False, fill_na)

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

    # Preprocessing the dirctors data
    #credits_dataframe['crew'] = credits_dataframe['crew'].apply(ast.literal_eval)
    #directors_data = credits_dataframe['crew'].apply(get_directors_ids)
    #credits_dataframe["directors"] = pd.Series(directors_data)

    # Print the shape of the dataframe
    print("  Credits dataframe shape:")
    print("  {0}".format(credits_dataframe.shape))

    return credits_dataframe

def extract_and_repair_data_from_new_file(imdb_movies_dataframe):
    """ Extracts and repairs the data from the imdb movies file """
    repaired_nrOfWins = []
    repaired_nrOfNominations = []
    repaired_nrOfPhotos = []
    repaired_nrOfNewsArticles = []
    repaired_nrOfUserReviews = []
    nrOfWins_items = imdb_movies_dataframe["nrOfWins"]
    for index, item in enumerate(nrOfWins_items):
        column = 10;
        try:
            if item.isdigit():
                if int(item) > 1000:
                    next_item = imdb_movies_dataframe.at[index, "nrOfNominations"]
                    if not next_item.isdigit():
                        column += 2;
                    else:
                        column += 3;
            else:
                column += 1;
            new_item = imdb_movies_dataframe.iat[index, column]
            repaired_nrOfWins.append(new_item)
            next_item = imdb_movies_dataframe.iat[index, column + 1]
            repaired_nrOfNominations.append(next_item)
            second_next_item = imdb_movies_dataframe.iat[index, column + 2]
            repaired_nrOfPhotos.append(second_next_item)
            third_next_item = imdb_movies_dataframe.iat[index, column + 3]
            repaired_nrOfNewsArticles.append(third_next_item)
            fourth_next_item = imdb_movies_dataframe.iat[index, column + 4]
            repaired_nrOfUserReviews.append(fourth_next_item)
        except:
            repaired_nrOfWins.append(np.nan)
            repaired_nrOfNominations.append(np.nan)
            repaired_nrOfPhotos.append(np.nan)
            repaired_nrOfNewsArticles.append(np.nan)
            repaired_nrOfUserReviews.append(np.nan)
    return (repaired_nrOfWins, repaired_nrOfNominations, repaired_nrOfPhotos,
           repaired_nrOfNewsArticles, repaired_nrOfUserReviews)

def add_new_columns_from_imdb_movies_dataframe(imdb_movies_dataframe, dataframe):
    """ Adds the the imdb movies data to the imdb_movies_dataframe """
    test_imdb = imdb_movies_dataframe[["tid", "nrOfWins", "nrOfNominations", "nrOfPhotos",
                                      "nrOfNewsArticles", "nrOfUserReviews"]]
    repaired_nrOfWins, repaired_nrOfNominations, repaired_nrOfPhotos, repaired_nrOfNewsArticles, repaired_nrOfUserReviews =\
       extract_and_repair_data_from_new_file(imdb_movies_dataframe)
            
    test_imdb["nrOfWins"] = edit_data_values(repaired_nrOfWins, zero_to_nan=False)
    test_imdb["nrOfNominations"] = edit_data_values(repaired_nrOfNominations, zero_to_nan=False)
    test_imdb["nrOfPhotos"] = edit_data_values(repaired_nrOfPhotos, zero_to_nan=False)
    test_imdb["nrOfNewsArticles"] = edit_data_values(repaired_nrOfNewsArticles, zero_to_nan=False)
    test_imdb["nrOfnrOfUserReviews"] = edit_data_values(repaired_nrOfUserReviews, zero_to_nan=False)

    dataframe = dataframe.copy().join(test_imdb)
    return dataframe

