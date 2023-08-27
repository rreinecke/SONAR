import pandas as pd

from build_tree import SONAR
from helpers import merge_dfs, create_folders


def get_input_data(path, model):
    df_p = pd.read_csv(path + "pr.csv")
    df_et = pd.read_csv(path + "domains.csv")
    df_et = df_et[["lat", "lon", "potevap_median"]]
    df_et.columns = ["lat", "lon", "PET"]

    df_days = pd.read_csv(path + "days_below_1.csv")
    df_days.rename(columns={'tas': '1d'}, inplace=True)

    df_temp = pd.read_csv(path + "tas.csv")
    df_temp["tas"] = df_temp["tas"] - 273.15

    df_lc = pd.read_csv("../Landcover/landcover.csv")
    lcc = "Forest", "Shrubland", "Grasland", "Sparsely Veg.", "Bare areas", "Wetland", "Cropland", "Waterbodies", \
        "SnowIce", "Artifical"
    df_lc['LC'] = pd.Categorical(df_lc['LC'], categories=lcc, ordered=False)

    df_r = pd.read_csv(path + model + "/qr.csv")
    # Remove negative values in PCR because they are likely an artifact of the cap.-rise
    # Remove 0 values because relatively unlikely over 30-year average
    df_r = df_r[df_r["qr"] > 0]

    l = [df_p, df_r, df_et, df_days, df_temp, df_lc]
    df = merge_dfs(l)
    df["Aridity"] = df["PET"] / df["pr"]
    df = df[["qr", "pr", "Aridity", "tas", "PET", "LC"]]
    # df = df[["qr","pr", "tas", "PET","LC"]]
    df = df.dropna()
    return df


# Path to the model data
path = "data/"
models = ["watergap2", "pcr-globwb", "clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro"]

create_folders(".", models)

for mod in models:
    print(f"Checking {mod}")
    df = get_input_data(path, mod)
    inputs = ["LC", "pr", "Aridity", "tas", "PET"]
    categoricals = ["LC"]

    sonar = SONAR(df, inputs, categoricals)
    # test all possible explanations for e.g. recharge
    sonar.prepare("qr")
    sonar.tree()

