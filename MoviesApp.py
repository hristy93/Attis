import sys
import numpy as np
import pandas as pd
import win_unicode_console

def enable_win_unicode_console():
    try:
        # Fix UTF8 output issues on Windows console.
        # Does nothing if package is not installed
        from win_unicode_console import enable
        enable()
    except ImportError:
        pass

def read_data(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe

def main():
    movies_metadata_file_path = "movies_metadata_test.csv"
    movies_metadata_dataframe = read_data(movies_metadata_file_path)

if __name__ == "__main__":
        # enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
