from build_tree import SONAR
from tree_comparison import calc_sampi, prototype_trees
from create_testdata import test_data
import pandas as pd


# Creation of the final data frame in which the results of the SONAR analysis are collected.
# The labels correspond to those of the Master's thesis (see Table 3.3).
Final_Table = pd.DataFrame({'Drittletzte (1)': [], 'Vorletzte (2)': [], 'Letzte (3)': [], 'Stärke (4)': [],
                            'Art (5)': [], 'Störvariablen (6)': [], 'Sampi_rel (7.1)': [], 'Sampi_abs (7.2)': [],
                            'Zähler': []})


r_list          = [0.1, 0.3, 0.5]                                    # Strength of the designated relationship
type_list       = ["lin", "squ", "cub", "exp", "srt", "sin", "ggk"]  # Type of the designated relationship
interf_var_list = [True, False]                                      # Presence of confounding variables in the data set
n = 1                                                                # Determining the global seed value


# For loop over all possible, unique specifications...
# ... of a data set (according to Table 3.3):
for a in ["X", "Kn", "Kt", "Y", "O"]:               # (1) Third last split
    for b in ["X", "Kn", "Kt", "Y", "O"]:           # (2) Penultimate split
        if (a != "O") and (b == "O"): continue
        if (a == b)   and (b != "O"): continue
        for c in ["X", "Kn", "Kt", "Y"]:            # (3) Last split
            if (c == a) or (c == b): continue
            if    (((c == "X") and (b == "Y") and (a != "O"))            # Would be possible, but could not be
                or ((c == "Y") and (b == "X") and (a != "O"))): continue #...implemented in a meaningful way within the
                                                                         # ...of this thesis.

            for d in r_list:                        # (4) Strength of the designated relationship
                for e in type_list:                 # (5) Type of the designated relationship
                    for f in interf_var_list:       # (6) Presence of confounding variables

                        # Get the test data set according to the unique specifications...
                        # ... to be able to apply SONAR to it:
                        test_df   = test_data(split1 = a, split2 = b, split3 = c, r_S= d,
                                              type = e, interf_var = f, seed = n)

                        # List of variables for the SONAR application:
                        variables = ["X", "Y"]
                        cat = []
                        if "Kn" in [a, b, c]: variables += ["Kn"]
                        if "Kt" in [a, b, c]:
                            variables += ["Kt"]; cat += ["Kt"]
                        if f:
                            variables += ["kog", "kon", "kat"]
                            cat += ["kat"]

                        # Application of SONARs for each unique test data set
                        sonar = SONAR(test_df, variables, categoricals=cat)
                        sonar.prepare("Y")
                        test_tree = sonar.tree()

                        # Get the prototypical tree based on the specifications of the test data set ...
                        # ... and compare the decision tree created by SONAR with the prototypical tree.
                        proto_tree = prototype_trees(var1 = a, var2 = b, var3 = c)
                        i_rel = calc_sampi(proto_tree, test_tree, type = "rel")
                        i_abs = calc_sampi(proto_tree, test_tree, type = "abs")

                        # Add the results to the list of all specifications:
                        Table_append = pd.DataFrame({'Drittletzte (1)': [a], 'Vorletzte (2)': [b],
                                                     'Letzte (3)': [c], 'Stärke (4)': [d], 'Art (5)': [e],
                                                     'Störvariablen (6)': [f],
                                                     'Sampi_rel (7.1)': [i_rel], 'Sampi_abs (7.2)': [i_abs],
                                                     'Zähler': [n]})
                        Table_append = Table_append.astype({'Zähler': str})
                        Table_append = Table_append.astype({'Störvariablen (6)': str})
                        Final_Table = pd.concat([Final_Table, Table_append], ignore_index=True)

                        # Set file name and save the updated file
                        dateiname = "Testbaumvergleich_Ergebnis.csv"
                        Final_Table.to_csv(dateiname, index=True)
                        n+=1 # Increase the global seed so that each data set is pseudo-random but unique.