from build_tree import SONAR
from tree_comparison import calc_sampi, prototype_trees
from create_testdata import test_data
import pandas as pd



Final_Table = pd.DataFrame({'Letzte (1)': [], 'Vorletzte (2)': [], 'Drittletzte (3)': [], 'Stärke (4)': [],
                            'Art (5)': [], 'Störvariablen (6)': [], 'Sampi_rel (7.1)': [], 'Sampi_abs (7.2)': []})


r_list = [0.1, 0.3, 0.5]
type_list = ["linear", "square", "cub", "exp", "exn", "sqrt"]
interf_var_list = [True, False]
n = 1
for a in ["O", "X", "Kn", "Kt", "Y"]:
    for b in ["O", "X", "Kn", "Kt", "Y"]:
        if (a != "O") and (b == "O"): continue
        if (a == b)   and (b != "O"): continue
        if ((a == "X") and (b == "Y")) or ((a == "Y") and (b == "X")): continue #Wäre zwar möglich, aber nicht sinnvoll umsetzbar.
        for c in ["X", "Kn", "Kt", "Y"]:
            if (c == a) or (c == b): continue
            for d in r_list:
                for e in type_list:
                    for f in interf_var_list:
                        print(f"split1 = {a}, split2 = {b}, split3 = {c}, pearson_r = {d}, type = {e}, interf. variables = {f}; counter = {n}")
                        test_df   = test_data(split1 = a, split2 = b, split3 = c, pearson_r = d, type = e, interf_var = f, seed = n)
                        variables = ["X", "Y"]
                        cat = []
                        if "Kn" in [a, b, c]: variables += ["Kn"]
                        if "Kt" in [a, b, c]:
                            variables += ["Kt"]; cat += ["Kt"]
                        if f:
                            variables += ["kog", "kon", "kat"]
                            cat += ["kat"]

                        sonar = SONAR(test_df, variables, categoricals=cat)
                        sonar.prepare("Y")
                        test_tree = sonar.tree()

                        proto_tree = prototype_trees(var1 = a, var2 = b, var3 = c)
                        i_rel = calc_sampi(proto_tree, test_tree, type = "rel")
                        i_abs = calc_sampi(proto_tree, test_tree, type = "abs")

                        Table_append = pd.DataFrame({'Letzte (1)': [c], 'Vorletzte (2)': [b],
                                                     'Drittletzte (3)': [a], 'Stärke (4)': [d], 'Art (5)': [e],
                                                     'Störvariablen (6)': [f],
                                                     'Sampi_rel (7.1)': [i_rel], 'Sampi_abs (7.2)': [i_abs]})
                        Table_append = Table_append.astype({'Störvariablen (6)': str})

                        print("\n Sampi_rel: ", i_rel, "// Sampi_abs: ", i_abs, "\n"
                                               "__________________________________________________ \n")

                        Final_Table = pd.concat([Final_Table, Table_append], ignore_index=True)

                        dateiname = "Testbaumvergleich_Ergebnis.csv"  # Dateiname festlegen
                        Final_Table.to_csv(dateiname, index=True)     # Abspeichern der Datei
                        n+=1