import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


mods = {"watergap2": {"leafs": [1,2], "splits": ["T <= 26°", "T > 26°C"]},
        "pcr-globwb": {"leafs": [1,2], "splits": ["AI <= .5", "AI > .5"]},
        "clm45": {"leafs": [1,3,4], "splits": ["AI <= .4", ".4 > AI <= .5", "AI > .5"]},
        "cwatm": {"leafs": [1,3,4], "splits": ["PET <=1300mm", "PET > 1300mm, T < 27°C", "PET > 1300mm, T > 27°C" ]},
        "h08": {"leafs": [1,2], "splits": ["T <= 27°", "T > 27°C"]},
        "jules-w1": {"leafs": [2,3,4], "splits": ["Sp. Veg., PET <= 380mm", "Sp. Veg., PET > 380mm", "Not Sp. Veg."]},
        "lpjml": {"leafs": [2,3,4], "splits": ["AI <= .4", ".5 > AI > .4" ,"AI > .5"]},
        "matsiro": {"leafs": [1,4,5,6], "splits": ["T <= 20°C", "T > 20°C, PET < 1300mm", "T > 20°C, 1400mm >= PET > 1300mm", "T > 20°C, PET > 1400mm"]},
        }

d = {"pr": "#8da0cb", "Aridity": "#fc8d62"}

def get_cutoffs(df):
    cats = ["very low", "low", "medium", "high"]
    df['cat'] = pd.cut(df['Y'], [0,10,100,500,2000], labels= cats)
    for c in cats:
        d = df[df['cat'] == c]
        print(c)
        print("Min: {} Max: {}".format(d["X"].min(), d["X"].max()))
    

def plotbinlines(x, qr):
    ax = plt.gca()
    n = 10
    bin_edges = stats.mstats.mquantiles(x[~np.isnan(x)], np.linspace(0, 1, n))
    median_stat = stats.binned_statistic(x, qr, statistic=np.nanmean, bins=bin_edges)
    y = median_stat.statistic
    min = stats.binned_statistic(x, qr, statistic=np.min, bins=bin_edges).statistic
    max = stats.binned_statistic(x, qr, statistic=np.max, bins=bin_edges).statistic
    yerr = np.vstack([y - min,max - y])
    bin_median = stats.mstats.mquantiles(x, np.linspace(0, 1, len(bin_edges)-1))
    line_data = np.stack([bin_median, y])
    df = pd.DataFrame(line_data.T, columns = ['X','Y'])
    get_cutoffs(df)

    ax.errorbar(bin_median, y, yerr=yerr, alpha=0.9, linestyle='solid', color='black', capsize=1, elinewidth=.5)

for model in mods:
    dfs = []
    print(model)
    for fi in os.listdir(model):
        l_id = int(fi.split('_')[0])
        leafs = mods[model]["leafs"]
        splits = mods[model]["splits"]
        if l_id in leafs:
            # it is a leaf
            df = pd.read_csv(os.path.join(model, fi))
            var = list(df.columns)
            df["Driver"] = var[0]
            if var[0] == "Aridity":
                # ignore all bigger AI values for plotting
                df = df[df["Aridity"] < 5]
            df.columns = ["x","qr", "Driver"]
            df["Leaf"] = l_id
            df["Domain"] = splits[leafs.index(l_id)]
            dfs.append(df)
    df = pd.concat(dfs)
    domains = df["Domain"].unique()
    for do in domains:
        data = df[df["Domain"] == do]
        plt.figure()
        s = sns.jointplot(data["x"], data["qr"], kind="hex", color=d[data["Driver"][0]], marginal_kws=dict(bins=100, fill=False), height=5, ratio=8, marginal_ticks=True)
        plotbinlines(data["x"], data["qr"]) 
        sns.despine(trim=True)
        plt.savefig(f"{model}_{do}.png", dpi=400)
        plt.clf()

