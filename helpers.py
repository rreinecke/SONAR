import os
import pandas as pd
from functools import reduce


def merge_dfs(ldf, on=None):
    """
    Turn a list of pandas data frames into one data frame along two given axis.
    @note The default is merging data on latitude and longitude.
     But this could also be different variables of time and space.
    """
    if on is None:
        on = ["lat", "lon"]
    return reduce(lambda x, y: pd.merge(x, y, on=on), ldf)


def write_data(dataframe, level, side, i, parent, x_name, y_name):
    """
    Write data of a current split to a CSV file.
    """
    d = dataframe[[x_name, y_name]]
    d.to_csv(f"{i}_tree_{level}_{side}_{parent}.csv", index=False)


def create_folders(path, mods):
    """
    Creates multiple folders for a list of models.
    """
    for model in mods:
        folder_path = os.path.join(path, model)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def iter_outputs(path):
    outputs = []
    for f in os.listdir(path):
        if f.endswith('.csv'):
            file_path = os.path.join(path, f)
            outputs.append(file_path)
    return outputs