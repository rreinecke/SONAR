from build_tree import SONAR
from tree_comparison import calc_sampi, prototype_trees
from create_testdata import test_data
import pandas as pd



Final_Table = pd.DataFrame({'Letzte (1)': [], 'Vorletzte (2)': [], 'Drittletzte (3)': [], 'Stärke (4)': [],
                            'Art (5)': [],
                            'StV kat (6.1)': [], 'StV kog (6.2)': [], 'StV kon (6.3)': [],
                            'Sampi_rel (7.1)': [], 'Sampi_abs (7.2)': []})


r_list = [0.1, 0.3, 0.5]
type_list = ["linear", "square", "cub", "exp", "exp_n", "sqrt"]
tf_list = [True, False]
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
                    for f in tf_list:
                        for g in tf_list:
                            for h in tf_list:
                                print(f"split1 = {a}, split2 = {b}, split3 = {c}, pearson_r = {d}, type = {e}, ord_rand = {f}, ord_norm = {g}, nom = {h}; counter = {n}")
                                test_df    = test_data(split1 = a, split2 = b, split3 = c, pearson_r = d, type = e, kat= f, kog= g, kon= h)

                                if "Kn" in [a, b, c]: variables = ["X", "Y", "Kn"]
                                else: variables = ["X", "Y"]
                                if "Kt" in [a, b, c]: cat = ["Kt"]
                                else: cat = []

                                sonar = SONAR(test_df, variables, categoricals=cat)
                                sonar.prepare("Y")
                                test_tree = sonar.tree()

                                proto_tree = prototype_trees(var1 = a, var2 = b, var3 = c)
                                i_rel = calc_sampi(proto_tree, test_tree, type = "rel")
                                i_abs = calc_sampi(proto_tree, test_tree, type = "abs")

                                Table_append = pd.DataFrame({'Letzte (1)': [c], 'Vorletzte (2)': [b],
                                                             'Drittletzte (3)': [a], 'Stärke (4)': [d], 'Art (5)': [e],
                                                             'StV kat (6.1)':[f],'StV kog (6.2)': [g],
                                                             'StV kon (6.3)': [h],
                                                             'Sampi_rel (7.1)': [i_rel], 'Sampi_abs (7.2)': [i_abs]})
                                Table_append = Table_append.astype({'StV kat (6.1)': str, 'StV kog (6.2)': str,
                                                                    'StV kon (6.3)': str})

                                print("\n Sampi_rel: ", i_rel, "// Sampi_abs: ", i_abs, "\n"
                                                       "__________________________________________________ \n")

                                Final_Table = pd.concat([Final_Table, Table_append], ignore_index=True)

                                dateiname = "Testbaumvergleich_Ergebnis.csv"  # Dateiname festlegen
                                Final_Table.to_csv(dateiname, index=True)     # Abspeichern der Datei
                                n+=1