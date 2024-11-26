import pandas as pd
import numpy as np
from scipy import stats
import os

# SONAR helper functions.
# The helper module contains smaller functions that do not directly describe the logic of SONAR

from helpers import write_data

import warnings

# Ignore warnings related to SpearmanRConstantInputWarning
warnings.filterwarnings("ignore")


def p_sym(p):  #Formatiert die p-Werte (Fehler 1. Art) für print-Befehle.
    if p >= 0.05:                     p_val = "= {:.3f}".format(p)
    elif (p < 0.05) and (p >= 0.01):  p_val = "< 0.05¹".format(p)
    elif (p < 0.01) and (p > 0.001):  p_val = "< 0.01²".format(p)
    elif (p < 0.001) and (p != -999): p_val = "< 0.001³".format(p)
    return p_val



class SONAR:
    """

    """

    # Global settings
    min_p = .05  # Minimum percentage of points of the previous parent node to be considered for a split.
    min_points = 500  # Minimum number of points to allow a split.
    alpha = 0.001  # Minimum p value of the correlation test to allow slit.
    bins = 100  # Number of bins to separate data into.

    def __init__(self, input_data, input_vars, categoricals=None, max_depth=None, type_rel = "spearman"):
        # Inputs supplied by the user
        if categoricals is None:
            categoricals = []
        self.inputs = input_vars  # Variables search space as list of strings
        if len(input_data[self.inputs[0]]) < self.min_points:
            print("It seems like you data has not enough points to allow for an analysis."
                  " You provided {} and the min is {}".format(len(input_data[self.inputs[0]]), self.min_points))
            exit()
        self.df = input_data  # Input data frame
        self.categoricals = categoricals  # List of categorical variables as list of strings
        self.max_depth = max_depth  # Trees are very small but one can also set a max depth, if none = disabled

        # Global variables
        self.target         = ""  # target variable
        self.splits         = []  # split decisions across the recursive calls
        self.u_id           = 0  # ID of the root node
        self.n_data         = len(self.df)  # total number of points
        self.actual_bins    = None  # Duplicated bins can lead to fewer bins than requested
        self.n_actual_bins  = 0
        self.tree_dict      = None      #Output-Tree
        if type_rel == "spearman":
            self.symbol     = "\u03f1"

    def prepare(self, target):
        """
        Prepare the SONAR algorithm with an initial correlation to the main search variable.
        The input 'var' needs to be contained in the input data
        """
        assert isinstance(target, str)
        self.target = target
        max_initial = 0
        max_initial_var = ""
        for i in self.inputs:
            if not i == self.target:
                if i not in self.categoricals:
                    c, _ = stats.spearmanr(self.df[target], self.df[i], axis=0)
                    if c > max_initial:
                        max_initial = c
                        max_initial_var = i
        print("Max initial correlation is {:.2f} to variable {}".format(max_initial, max_initial_var))
        self.tree_dict = {'Corr': max_initial, 'Relationship_Var' : max_initial_var, 'DP': len(self.df[self.inputs[0]])}

        # Initialize cutting points
        for target in self.inputs:
            if target in self.categoricals:
                continue
            # df[var + "_cat"] = pd.cut(df[var], bins=bins, include_lowest=True, ordered=True)
            # df[var + "_code"] = pd.cut(df[var], bins=bins, include_lowest=True, ordered=True,
            # labels=list(range(0, bins)))
            self.df[target + "_cat"], self.actual_bins = pd.qcut(self.df[target], q=self.bins,
                                                                 duplicates="drop", retbins=True)
            self.df[target + "_code"] = pd.qcut(self.df[target], q=self.bins,
                                                labels=list(range(0, len(self.actual_bins) - 1)), duplicates="drop")
            self.df[target + "_code"] = self.df[target + "_code"].astype('int')

        # If we have duplicated bins due to qcut we can have fewer bins than requested
        self.n_actual_bins = len(self.actual_bins)

    def tree(self):
        """
        Wrapper call to build tree.
        """
        return self.get_tree(self.df, self.max_depth, self.inputs, self.n_actual_bins)

    def check_split(self, bucket_l, bucket_r, base_corr, cat, lvar, max_corr, categorical, cat_ranges=None):
        """
        This method implements the split decision.
        """
        split_info = {"max_corr":        0,  # All the information recorded for a specific split
                      "max_val":         0,
                      "max_cat":         "",
                      "is_cat_split":    True,
                      "max_var":         "",
                      "leri":            "left",
                      "relationship_var": "",
                      "success":          False,
                      "max_corr_l":       0,
                      "max_corr_r":       0,
                      "min_p_val_l":      1,
                      "min_p_val_r":      1}

        # The left and right bucket of the current split
        n_lbucket = len(bucket_l)
        n_rbucket = len(bucket_r)

        # check if both buckets would be big enough
        if (n_lbucket >= self.min_points and n_lbucket / self.n_data > self.min_p and
                n_rbucket >= self.min_points and n_rbucket / self.n_data > self.min_p):
            for rel in self.inputs:
                if rel in self.categoricals:
                    continue
                corr_l = 0
                corr_r = 0

                # Iterate all possible relationships for this split. The p-value roughly indicates the probability of
                # an uncorrelated system producing datasets that have a Spearman correlation at least as extreme as
                # the one computed from these datasets. Although calculation of the p-value does not make strong
                # assumptions about the distributions underlying the samples, it is only accurate for very large
                # samples (>500 observations).

                # Left
                corr_tmp, p_value = stats.spearmanr(bucket_l[rel], bucket_l[self.target], axis=0)
                if p_value < self.alpha and corr_tmp != np.nan:
                    corr_l  = np.abs(corr_tmp)
                    p_val_l = p_value

                # Right
                corr_tmp, p_value = stats.spearmanr(bucket_r[rel], bucket_r[self.target], axis=0)
                if p_value < self.alpha and corr_tmp != np.nan:
                    corr_r  = np.abs(corr_tmp)
                    p_val_r = p_value

                if corr_r == 0 and corr_l == 0:
                    continue

                if corr_r < base_corr and corr_l < base_corr:
                    # split is not improving the correlation since last split
                    continue

                # is the current correlation a new maximum? If yes save position as potential split
                split_found = ""
                if max_corr < corr_l:
                    split_found = "left"
                if max_corr < corr_r:
                    split_found = "right"

                if split_found:
                    if not categorical:
                        # Calculate the split position
                        centroid = 0
                        # One could also use the .mid of pandas
                        centroid = cat_ranges[cat].left + ((cat_ranges[cat].right - cat_ranges[cat].left) / 2)
                        split_info["max_val"] = centroid
                    else:
                        split_info["max_val"] = cat

                    if split_found == "left":
                        split_info["max_corr"] = np.abs(corr_l)
                    else:
                        split_info["max_corr"] = np.abs(corr_r)

                    split_info["max_cat"] = cat
                    split_info["is_cat_split"] = categorical
                    split_info["max_var"] = lvar
                    split_info["leri"] = split_found
                    split_info["relationship_var"] = rel
                    split_info["success"] = True
                    split_info["max_corr_l"]  = corr_l
                    split_info["max_corr_r"]  = corr_r
                    split_info["p_val_l"] = p_sym(p_val_l)
                    split_info["p_val_r"] = p_sym(p_val_r)
        return split_info

    def get_tree(self, data, max_depth, l_var, n_bins, cd=0, id=None, base_corr=0, tree_dict = None):
        """
        l_vars: list of variables to test
        n_bins: number of bins
        base_corr: correlation of last split
        """

        # print("Tree at depth: {}".format(cd))
        if cd == max_depth:
            return self.tree_dict
        spaces = ""
        if id is None:
            # plot_root(data)
            id = 0
            print("Root node")

        if not tree_dict:
            tree_dict = self.tree_dict

        max_corr = 0  # biggest correlation
        max_corr_l = 0  # left corr at max
        max_corr_r = 0  # right corr at max
        max_val = 0  # split point value
        max_cat = 0  # split point category
        is_cat_split = False
        max_var = ""
        leri = ""
        relationship_var = ""  # what relationship did we look at for the split
        success = False  # have we found any split?

        for j in range(0, len(l_var)):
            # iterate over all inputs
            # print("Testing {}".format(l_var[j]))
            if l_var[j] in self.categoricals:
                # no binning on categoricals
                cats = data[l_var[j]].cat.categories
                for c in range(0, len(cats)):
                    # iterate over all categories and test if they split the data with a high correlation
                    bucket_l = data.loc[data[l_var[j]] == cats[c]]
                    bucket_r = data.loc[data[l_var[j]] != cats[c]]
                    res = self.check_split(bucket_l, bucket_r, base_corr, cats[c], l_var[j], max_corr, True)
                    if not res["success"]:
                        continue
                    else:
                        success = True
                        max_corr = res["max_corr"]      # biggest correlation
                        max_corr_l = res["max_corr_l"]  # left corr at max
                        max_corr_r = res["max_corr_r"]  # right corr at max
                        p_val_l = res["p_val_l"]        # left p_value
                        p_val_r = res["p_val_r"]        # right p_value
                        max_val = res["max_val"]        # split point value
                        max_cat = res["max_cat"]        # split point category
                        is_cat_split = res["is_cat_split"]
                        max_var = res["max_var"]
                        leri = res["leri"]
                        relationship_var = res["relationship_var"]  # what relationship did we look at for the split
            else:
                cat_ranges = data[l_var[j] + "_cat"].cat.categories

                for i in range(0, n_bins):
                    # iterate over bins in input

                    # Pop a bin to the left bucket
                    bucket_l = data.loc[data[l_var[j] + "_code"] <= i]

                    # copy rest to right bucket -> likely inefficient
                    bucket_r = data.loc[data[l_var[j] + "_code"] > i]

                    res = self.check_split(bucket_l, bucket_r, base_corr, i, l_var[j], max_corr, False, cat_ranges)

                    if not res["success"]:
                        continue
                    else:
                        success = True
                        max_corr = res["max_corr"]      # biggest correlation
                        max_corr_l = res["max_corr_l"]  # left corr at max
                        max_corr_r = res["max_corr_r"]  # right corr at max
                        p_val_l = res["p_val_l"]        # left p_value
                        p_val_r = res["p_val_r"]        # right p_value
                        max_val = res["max_val"]        # split point value
                        max_cat = res["max_cat"]        # split point category
                        is_cat_split = res["is_cat_split"]
                        max_var = res["max_var"]
                        leri = res["leri"]
                        relationship_var = res["relationship_var"]  # what relationship did we look at for the split

                for i in range(n_bins - 1, -1, -1):
                    # iterate over bins in reverse

                    # Pop a bin to the left bucket
                    bucket_l = data.loc[data[l_var[j] + "_code"] <= i]

                    # copy rest to right bucket -> likely inefficient
                    bucket_r = data.loc[data[l_var[j] + "_code"] > i]

                    res = self.check_split(bucket_l, bucket_r, base_corr, i, l_var[j], max_corr, False, cat_ranges)

                    if not res["success"]:
                        continue
                    else:
                        success = True
                        max_corr = res["max_corr"]  # biggest correlation
                        max_corr_l = res["max_corr_l"]  # left corr at max
                        max_corr_r = res["max_corr_r"]  # right corr at max
                        p_val_l = res["p_val_l"]    # left p_value
                        p_val_r = res["p_val_r"]    # right p_value
                        max_val = res["max_val"]  # split point value
                        max_cat = res["max_cat"]  # split point category
                        is_cat_split = res["is_cat_split"]
                        max_var = res["max_var"]
                        leri = res["leri"]
                        relationship_var = res["relationship_var"]  # what relationship did we look at for the split

        # have we found a split?
        if not success:
            return self.tree_dict

        # We have found a better split
        self.splits.append(
            {"Variable": max_var, "Position": max_val, "Depth": cd, "Correlation": max_corr, "Maximized": leri,
             "Relationship": relationship_var})

        # Keep splitting
        if is_cat_split:
            data_l = data.loc[data[max_var] == max_cat].copy()
            data_r = data.loc[data[max_var] != max_cat].copy()
            split_val = max_cat
        else:
            data_l = data.loc[data[max_var + "_code"] <= max_cat].copy()
            data_r = data.loc[data[max_var + "_code"] > max_cat].copy()
            split_val = max_val

        tree_dict['Node'] = {'Split': True, 'Split_Var': max_var, 'Split_Value': split_val}
        tree_dict['left'] = {'Corr': max_corr_l, 'Relationship_Var': relationship_var, 'DP': len(data_l), 'Node': {'Split': False}}
        tree_dict['right']= {'Corr': max_corr_r, 'Relationship_Var': relationship_var, 'DP': len(data_r), 'Node': {'Split': False}}

        self.u_id += 1

        spaces += '--' * cd
        if is_cat_split:
            print("|-" + spaces + "  {}: {} == {} with {} points; {} = {:.2f} (p = {}); dri. = {}".format(self.u_id, max_var, max_val,
                                                                                                len(data_l), self.symbol,
                                                                                                 max_corr_l, p_val_l,
                                                                                                relationship_var))
        else:
            print("|-" + spaces + "  {}: {} <= {:.2f} with {} points; {} = {:.2f} (p = {}); dri. = {}".format(self.u_id, max_var,
                                                                                                    max_val,
                                                                                                    len(data_l), self.symbol,
                                                                                                    max_corr_l, p_val_l,
                                                                                                    relationship_var))
        # print("-> LEFT")
        # Left tree
        if len(data_l) > 0:
            write_data(data_l, cd + 1, "left", self.u_id, id, relationship_var, self.target)
            self.get_tree(data_l, max_depth, l_var, n_bins, cd + 1, self.u_id, max_corr, tree_dict['left'])

        self.u_id += 1

        if is_cat_split:
            print("|-" + spaces + "  {}: {} != {} with {} points; {} = {:.2f} (p = {}); dri. = {}".format(self.u_id, max_var, max_val,
                                                                                                len(data_r), self.symbol,
                                                                                                 max_corr_r, p_val_r,
                                                                                                relationship_var))
        else:
            print(
                "|-" + spaces + "  {}: {} > {:.2f} with {} points; {} = {:.2f} (p = {}); dri. = {}".format(self.u_id, max_var, max_val,
                                                                                                 len(data_r), self.symbol,
                                                                                                 max_corr_r, p_val_r,
                                                                                                 relationship_var))
        # print("-> RIGHT")
        # Right tree
        if len(data_r) > 0:
            write_data(data_r, cd + 1, "right", self.u_id, id, relationship_var, self.target)
            self.get_tree(data_r, max_depth, l_var, n_bins, cd + 1, self.u_id, max_corr, tree_dict['right'])

        if cd == 0:
            return self.tree_dict