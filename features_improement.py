import numpy as np
import pandas as pd

from utils import *

def process_keywords(keywords_dataframe):
    bags_of_keywords = [kvp_words
       for bag_of_words in keywords_dataframe["keywords"]
           for kvp_words in ast.literal_eval(bag_of_words)]
    count_of_all = len(bags_of_keywords)
    unique_keywords = { kvp_words["id"]: kvp_words["name"].replace(" ", "_")
                      for kvp_words in bags_of_keywords }
    count_of_unique = len(unique_keywords)

    #keywords = [keyword for keyword in unique_keywords.values()]



    print(unique_keywords)
