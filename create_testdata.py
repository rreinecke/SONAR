import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import build_tree


depth_faktor = 2/3

#print(stats.pearsonr(df["Xi"], df["Y"]))
#print(stats.pearsonr(df["Xi"][df["Xt"] == "C"], df["Y"][df["Xt"] == "C"]))
#print(stats.pearsonr(df["Xi"][df["Y"] < 0.5], df["Y"][df["Y"] < 0.5]))

def print_dataset(df):
    df_plt_obj = sns.pairplot(df, diag_kind="kde", plot_kws=dict(marker="x", alpha = 0.9, linewidth=0.08))
    #df_plt_obj.map_lower(sns.kdeplot, levels=4, color="0.2")
    plt.show()


#____________________________________________________________
#____________________________________________________________


def create_dataset(pearson_r, size=1000, xlim_min = 0, xlim_max =1, depth = 1):
    m = pearson_r*0.6* ((depth_faktor)**(depth-1))
    n = 0.5-0.5*m
    alpha = 0.001
    x1 = np.arange(1, size + 1)
    p_value, corr, exit_p = 1, -999, 0
    while True:
        exit_p = exit_p + 1
        x2_independent = np.random.normal(0, 1, size)
        x2 = (pearson_r * (x1 - np.mean(x1)) / np.std(x1) + np.sqrt(1 - pearson_r ** 2) * x2_independent)
        a, b = np.polyfit(x1, x2, 1)   #identifizieren der Regressionsgeraden y = a*x+b
        x2 = ((x2 - b) / a) / size           # normieren auf y = 1*x + 0
        x2 = x2 *m*(xlim_max-xlim_min)      #Anpassen des Steigungsfaktors an die gewünschte Form
        x2 = x2 + n + m*xlim_min            #Anpassen des y-Achsen Abschnitts an die gewünschte Form
        for i in range(len(x2)):
            if (x2[i] <= 0) or (x2[i] >1):
                x2[i] =np.random.randint(1,10001)/10000

        corr, p_value = stats.pearsonr(x1, x2)
        # Checkt, ob alle Bedingungen erfüllt sind.
        if (p_value <= alpha) & (abs(corr - pearson_r) < 0.01):
            break
        # Abbruchbedingung, falls kein Ergebnis erreicht wird.
        if exit_p >= 1000:
            print("Requirements not satisfied")
            return
    # Erzeugt ein Pandas DF
    df = pd.DataFrame({'Xi': (x1*(xlim_max-xlim_min)/size) + xlim_min, 'Y':x2})
    return df


#____________________________________________________________
#____________________________________________________________


def create_X(pearson_r, printit = False):
    df = create_dataset(pearson_r = pearson_r)
    df["Xi"] = df["Xi"]/10

    x1_rand = np.random.randint(1001,10001, 9000)/10000
    x2_rand = np.random.randint(1,10001, 9000)/10000
    df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand})
    df = pd.concat([df, df_append], ignore_index= True)
    if printit: print_dataset(df)
    return df

#df = create_X(0.1, printit = True)

def create_K(pearson_r, printit = False):
    df = create_dataset(pearson_r=pearson_r)
    x3_df = np.random.randint(1,1001, 1000)/10000
    df["Xk"] =x3_df

    x1_rand = np.random.randint(1,10001, 9000)/10000
    x2_rand = np.random.randint(1,10001, 9000)/10000
    x3_rand = np.random.randint(1001,10001, 9000)/10000
    df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, "Xk": x3_rand})
    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#df = create_K(0.99, printit = True)

def create_T(pearson_r, printit = False):
    df = create_dataset(pearson_r=pearson_r)
    x3_df =  ["A"] * 1000
    df["Xt"] = x3_df

    x1_rand = np.random.randint(1,10001, 9000)/10000
    x2_rand = np.random.randint(1,10001, 9000)/10000
    x3_rand = np.random.choice(["B", "C", "D", "E"], 9000)

    df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, "Xt": x3_rand})
    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#df = create_T(0.9, printit = True)

def create_Y(pearson_r, printit = False):
    df = create_dataset(pearson_r=pearson_r)
    Xi = df["Y"]
    Y  = df["Xi"] / 10
    df = pd.DataFrame({'Xi': Xi, 'Y': Y})
    x1_rand = np.random.randint(1,10001, 9000)/10000
    x2_rand = np.random.randint(1001,10001, 9000)/10000
    df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand})
    df = pd.concat([df, df_append], ignore_index= True)
    if printit: print_dataset(df)
    return df

#df = create_Y(0.99, printit = True)


#____________________________________________________________
#____________________________________________________________


def create_var1_X(pearson_r, var1, printit = False):
    df_sign = create_dataset(pearson_r=pearson_r,     size=1000, xlim_min = 0, xlim_max = 0.1, depth = 2)
    df_rest = create_dataset(pearson_r=pearson_r*depth_faktor, size=9000, xlim_min = 0.1, xlim_max = 1, depth = 1)
    df = pd.concat([df_sign, df_rest], ignore_index=True)

    if var1 == "K":
        x3 = np.random.randint(1,5001, 10000)/10000
        df["Xk"] = x3
        x1_rand = np.random.randint(1,    10001, 10000)/10000
        x2_rand = np.random.randint(1,    10001, 10000)/10000
        x3_rand = np.random.randint(5001, 10001, 10000)/10000
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})

    elif var1 == "T":
        x3 =  ["B"] * 10000
        df["Xt"] = x3
        x1_rand = np.random.randint(1, 10001, 10000) / 10000
        x2_rand = np.random.randint(1, 10001, 10000) / 10000
        x3_rand = np.random.choice(["A", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand})

    elif var1 == "Y":
        df["Y"] = df["Y"]/2
        x1_rand = np.random.randint(1,   10001, 10000)/10000
        x2_rand = np.random.randint(5001,10001, 10000)/10000
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand})

    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_X(1, "K", printit = True)
#create_var1_X(1, "T", printit = True)
#create_var1_X(0.8, "Y", printit = True)

def create_var1_K(pearson_r, var1, printit = False):
    df = create_dataset(pearson_r=pearson_r, size = 1000, depth = 2)
    x3_df = np.random.randint(1,1001, 1000)/10000
    df["Xk"] =x3_df
    df_append = create_dataset(pearson_r=pearson_r*depth_faktor, size = 9000)
    x3_append = np.random.randint(1001, 10001, 9000)/10000
    df_append["Xk"] =x3_append
    df = pd.concat([df, df_append], ignore_index=True)

    if var1 == "S":
        df["Xi"] /=2
        x1_rand = np.random.randint(5001,10001, 10000) / 10000
        x2_rand = np.random.randint(1,   10001, 10000) / 10000
        x3_rand = np.random.randint(1,   10001, 10000) / 10000
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})
    elif var1 == "T":
        x4 = ["B"] * 10000
        df["Xt"] = x4
        x1_rand = np.random.randint(1,10001, 10000) / 10000
        x2_rand = np.random.randint(1,10001, 10000) / 10000
        x3_rand = np.random.randint(1,10001, 10000) / 10000
        x4_rand = np.random.choice(["A", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand, 'X4': x4_rand})
    elif var1 == "Y":
        df["Y"] /= 2
        x1_rand = np.random.randint(1,   10001, 10000) / 10000
        x2_rand = np.random.randint(5001,10001, 10000) / 10000
        x3_rand = np.random.randint(1,   10001, 10000) / 10000
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})
    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_K(0.9,"X", printit = True)
#create_var1_K(1,"T", printit = True)
#create_var1_K(1,"Y", printit = True)

def create_var1_T(pearson_r, var1, printit = False):
    df = create_dataset(pearson_r=pearson_r, size = 1000, depth = 2)
    x3 = ["A"] * 1000
    df['Xt'] = x3
    df_append = create_dataset(pearson_r=pearson_r*depth_faktor, size = 9000)
    x3_append = np.random.choice(["B", "C", "D", "E"])
    df_append['Xt'] = x3_append
    df = pd.concat([df, df_append], ignore_index=True)
    if var1 == "X":
        df['Xi'] /= 2
        x1_rand = np.random.randint(5001,10001, 10000) / 10000
        x2_rand = np.random.randint(1,   10001, 10000) / 10000
        x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand})
    if var1 == "K":
        x4_append = np.random.randint(1,    5001, 10000) / 10000
        df['Xk'] = x4_append
        x1_rand = np.random.randint(1,   10001, 10000) / 10000
        x2_rand = np.random.randint(1,   10001, 10000) / 10000
        x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        x4_rand = np.random.randint(5001,10001, 10000) / 10000
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand, 'Xk':x4_rand})
    if var1 == "Y":
        df['Y'] /= 2
        x1_rand = np.random.randint(1,   10001, 10000) / 10000
        x2_rand = np.random.randint(5001,10001, 10000) / 10000
        x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand})
    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)

#create_var1_T(.1,"X",True)
#create_var1_T(1,"K",True)
#create_var1_T(1,"Y",True)

def create_var1_Y(pearson_r, var1, printit = False):
    df = create_dataset(pearson_r=pearson_r, size = 1000, xlim_min = 0, xlim_max = 0.1, depth = 2)
    Xi, Y = df["Y"], df["Xi"]
    df["Xi"], df["Y"] = Xi, Y
    df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size = 9000, xlim_min = 0.1, xlim_max = 1, depth = 1)
    Xi, Y = df_append["Y"], df_append["Xi"]
    df_append["Xi"], df_append["Y"] = Xi, Y
    df = pd.concat([df, df_append], ignore_index= True)
    if var1 == "X":
        df["Xi"] /= 2
        x1_rand = np.random.randint(5001,10001, 10000) / 10000
        x2_rand = np.random.randint(1   ,10001, 10000) / 10000
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand})
    if var1 == "K":
        x3_append = np.random.randint(1,5001, 10000) / 10000
        df["Xk"] = x3_append
        x1_rand = np.random.randint(1,   10001, 10000) / 10000
        x2_rand = np.random.randint(1,   10001, 10000) / 10000
        x3_rand = np.random.randint(5001,10001, 10000) / 10000
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk':x3_rand})
    if var1 == "T":
        x3_append = ["B"] * 10000
        df['Xt'] = x3_append
        x1_rand = np.random.randint(1,   10001, 10000) / 10000
        x2_rand = np.random.randint(1,   10001, 10000) / 10000
        x3_rand = np.random.choice(["A", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt':x3_rand})
    df = pd.concat([df, df_append], ignore_index= True)
    if printit: print_dataset(df)
    return df

#create_var1_Y(.1, "X", printit = True)
#create_var1_Y(.1, "K", printit = True)
#create_var1_Y(.1, "T", printit = True)



#____________________________________________________________
#____________________________________________________________

def create_var1_var2_X(pearson_r, var1, var2, printit = False):
    df_sign = create_dataset(pearson_r=pearson_r,             size=1000, xlim_min = 0, xlim_max = 0.1, depth = 3)
    df_rest = create_dataset(pearson_r=pearson_r*depth_faktor,size=9000, xlim_min = 0.1, xlim_max = 1, depth = 2)
    df = pd.concat([df_sign, df_rest], ignore_index=True)

    if var2 == "T": #var3 = X
        x3 =  ["B"] * 10000
        df["Xt"] = x3
        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**5),size=10000, xlim_min = 0, xlim_max = 1, depth = 1)
        x3_append = np.random.choice(["A", "C", "D", "E"], 10000)
        df_append['Xt'] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "K": #var2 = T, var3 = X
            x4 = np.random.randint(1,   5001, 20000) / 10000
            df['Xk'] = x4
            x1_rand = np.random.randint(1,    10001, 20000) / 10000
            x2_rand = np.random.randint(1,    10001, 20000) / 10000
            x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            x4_rand = np.random.randint(5001, 10001, 20000) / 10000
            df_append = pd.DataFrame({'Xi':x1_rand, 'Y': x2_rand, 'Xt': x3_rand, 'Xk': x4_rand})

        elif var1 == "Y": #var2 = T, var3 = X
            df["Y"] /= 2
            x1_rand = np.random.randint(1,   10001, 20000) / 10000
            x2_rand = np.random.randint(5001,10001, 20000) / 10000
            x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi':x1_rand, 'Y': x2_rand, 'Xt': x3_rand})

    elif var2 == "K": #var3 = X
        x3 = np.random.randint(1,5001, 10000)/10000
        df["Xk"] = x3
        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2),size=10000, xlim_min = 0, xlim_max = 1, depth = 1)
        x3_append = np.random.randint(5001, 10001, 10000)/10000
        df_append['Xk'] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "T": #var2 = K, var3 = X
            x4 = ["C"]*20000
            df['Xt'] = x4
            x1_rand = np.random.randint(1,10001, 20000)/10000
            x2_rand = np.random.randint(1,10001, 20000)/10000
            x3_rand = np.random.randint(1,10001, 20000)/10000
            x4_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi':x1_rand, 'Y': x2_rand, 'Xk': x3_rand, 'Xt': x4_rand})

        elif var1 == "Y": #var2 = K, var3 = X
            df["Y"] /= 2
            x1_rand = np.random.randint(1,   10001, 20000)/10000
            x2_rand = np.random.randint(5001,10001, 20000)/10000
            x3_rand = np.random.randint(1,   10001, 20000)/10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})

    elif var2 == "Y": #var3 = X
        df["Y"]  = df["Y"]/2
        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2),size=10000, xlim_min = 0, xlim_max = 1, depth = 1)
        df_append['Y'] = df_append['Y']/2 + 0.5
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "K": #var2 = Y, var3 = X
            x3 = np.random.randint(1,5001, 20000) / 10000
            df['Xk'] = x3
            x1_rand = np.random.randint(1,   10001, 20000)/10000
            x2_rand = np.random.randint(1,   10001, 20000)/10000
            x3_rand = np.random.randint(5001,10001, 20000)/10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})

        if var1 == "T": #var2 = Y, var3 = X
            x3 = ["C"] * 20000
            df['Xt'] = x3
            x1_rand = np.random.randint(1,   10001, 20000)/10000
            x2_rand = np.random.randint(1,   10001, 20000)/10000
            x3_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})

    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_var2_X(1,"K", "T", printit = True)
#create_var1_var2_X(1,"Y", "T", printit = True)
#create_var1_var2_X(1,"T", "K", printit = True)
#create_var1_var2_X(1,"Y", "K", printit = True)
#create_var1_var2_X(1,"K", "Y", printit = True)
#create_var1_var2_X(1,"T", "Y", printit = True)

def create_var1_var2_K(pearson_r, var1, var2, printit = False):

    df = create_dataset(pearson_r=pearson_r, size = 1000, depth = 3)
    x3_df = np.random.randint(1,1001, 1000)/10000
    df["Xk"] =x3_df
    df_append = create_dataset(pearson_r=pearson_r*depth_faktor, size = 9000, depth = 2)
    x3_append = np.random.randint(1001, 10001, 9000)/10000
    df_append["Xk"] =x3_append
    df = pd.concat([df, df_append], ignore_index=True)


    if var2 == "X": #var3 = K     CAVE: Hier neuer Datensatz, um die Limits für den x-Split an die Steigung des späteren k-Splits anzupassen
        df = create_dataset(pearson_r=pearson_r, size=1000, depth=3, xlim_min = 0, xlim_max = 0.5)
        x3_df = np.random.randint(1, 1001, 1000) / 10000
        df["Xk"] = x3_df
        df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size=9000, depth=2, xlim_min = 0, xlim_max = 0.5)
        x3_append = np.random.randint(1001, 10001, 9000) / 10000
        df_append["Xk"] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        #Datensatz für den X-Split
        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2),size=10000, xlim_min = 0.5, xlim_max = 1, depth = 1)
        x3_rand = np.random.randint(1,   10001, 10000) / 10000
        df_append["Xk"] = x3_rand
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "T": #var2 = X var3 = K
            df["Xt"] =  ["C"] * 20000
            x1_rand = np.random.randint(1,   10001, 20000)/10000
            x2_rand = np.random.randint(1,   10001, 20000)/10000
            x3_rand = np.random.randint(1,   10001, 20000)/10000
            x4_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi':x1_rand, 'Y':x2_rand, 'Xk': x3_rand, 'Xt':x4_rand})

        if var1 == "Y": #var2 = X var3 = K
            df["Y"] /= 2
            x1_rand = np.random.randint(1,   10001, 20000)/10000
            x2_rand = np.random.randint(5001,10001, 20000)/10000
            x3_rand = np.random.randint(1,   10001, 20000)/10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})

    elif var2 == "T": #var3 = K
        x4 = ["B"] * 10000
        df["Xt"] = x4
        df_append = create_dataset(pearson_r = pearson_r*(depth_faktor**2), size = 10000, xlim_min = 0, xlim_max = 1, depth = 1)
        df_append['Xk'] = np.random.randint(1,10001, 10000) / 10000
        df_append['Xt'] = np.random.choice(["A", "C", "D", "E"], 10000)
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = T, var3 = K
            df["Xi"] /= 2
            x1_rand = np.random.randint(5001,10001, 20000) / 10000
            x2_rand = np.random.randint(1,   10001, 20000) / 10000
            x3_rand = np.random.randint(1,   10001, 20000) / 10000
            x4_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand, 'Xt': x4_rand})

        if var1 == "Y": #var2 = T, var3 = K
            df["Y"] /= 2
            x1_rand = np.random.randint(1,   10001, 20000) / 10000
            x2_rand = np.random.randint(5001,10001, 20000) / 10000
            x3_rand = np.random.randint(1,   10001, 20000) / 10000
            x4_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand, 'Xt': x4_rand})

    elif var2 == "Y": #var3 = K     | || CAVE: Hier neuer Datensatz, um die Limits für den x-Split an die Steigung des späteren k-Splits anzupassen
                      #                        Außerdem Index-Shift (Xi und Y vertauscht, um nach vor dem Split in Y noch einen sinnvollen Split einzubauen.
        #Neuer Datensatz (Eigentlich wie der "alte" nur mit veränderten Limits aufgrund der Steigung)
        df = create_dataset(pearson_r=pearson_r, size=1000, depth=3, xlim_min = 0, xlim_max = 0.5)
        x3_df = np.random.randint(1, 1001, 1000) / 10000
        df["Xk"] = x3_df
        df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size=9000, depth=2, xlim_min = 0, xlim_max = 0.5)
        x3_append = np.random.randint(1001, 10001, 9000) / 10000
        df_append["Xk"] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        #Index-Shift
        Xi, Y = df["Y"], df["Xi"]
        df["Xi"], df["Y"] = Xi, Y
        df_append = create_dataset(pearson_r = pearson_r*(depth_faktor**2), size = 10000, xlim_min = 0.5, xlim_max = 1, depth = 1)
        Xi, Y = df_append["Y"], df_append["Xi"]
        df_append["Xi"], df_append["Y"] = Xi, Y
        df_append["Xk"] = np.random.randint(1,   10001, 10000) / 10000
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = Y, var3 = K
            df['Xi'] /= 2
            x1_rand = np.random.randint(5001,10001, 20000) / 10000
            x2_rand = np.random.randint(1,   10001, 20000) / 10000
            x3_rand = np.random.randint(1,   10001, 20000) / 10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})

        elif var1 == "T": #var2 = Y, var3 = K
            df["Xt"] = ["C"] * 20000
            x1_rand = np.random.randint(1,   10001, 20000) / 10000
            x2_rand = np.random.randint(1,   10001, 20000) / 10000
            x3_rand = np.random.randint(1,   10001, 20000) / 10000
            x4_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand, 'Xt': x4_rand})

    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_var2_K(pearson_r = 1, var1 = "T", var2 = "X", printit = True)
#create_var1_var2_K(pearson_r = 1, var1 = "Y", var2 = "X", printit = True)
#create_var1_var2_K(pearson_r = 1, var1 = "X", var2 = "T", printit = True)
#create_var1_var2_K(pearson_r = 1, var1 = "Y", var2 = "T", printit = True)
#create_var1_var2_K(pearson_r = 1, var1 = "X", var2 = "Y", printit = True)
#create_var1_var2_K(pearson_r = 1, var1 = "T", var2 = "Y", printit = True)

def create_var1_var2_T(pearson_r, var1, var2, printit = False):
    df = create_dataset(pearson_r=pearson_r, size = 1000, depth = 3, xlim_min = 0)
    x3 = ["A"] * 1000
    df['Xt'] = x3
    df_append = create_dataset(pearson_r=pearson_r*depth_faktor, size = 9000, depth = 2)
    x3_append = np.random.choice(["B", "C", "D", "E"])
    df_append['Xt'] = x3_append
    df = pd.concat([df, df_append], ignore_index=True)

    if var2 == "X": #var3 = T     CAVE: Hier neuer Datensatz, um die Limits für den x-Split an die Steigung des späteren t-Splits anzupassen
        df = create_dataset(pearson_r=pearson_r, size=1000, xlim_min = 0, xlim_max = 0.5, depth=3)
        x3 = ["A"] * 1000
        df['Xt'] = x3
        df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size=9000, xlim_max = 0.5, depth = 2)
        x3_append = np.random.choice(["B", "C", "D", "E"])
        df_append['Xt'] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2), size=10000, xlim_min = 0.5, xlim_max = 1, depth=1)
        df_append['Xt']  = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "K": #var2 = X, var3 = T
            df["Xk"] = np.random.randint(1,    5001,20000) / 10000
            x1_rand= np.random.randint(1,   10001,20000) / 10000
            x2_rand= np.random.randint(1,   10001,20000) / 10000
            x3_rand= np.random.choice(["A", "B", "C", "D", "E"], 20000)
            x4_rand= np.random.randint(5001,10001,20000) / 10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand, 'Xk': x4_rand})

        elif var1 == "Y": #var2 = X, var3 = T
            df['Y'] /= 2
            x1_rand  = np.random.randint(1,   10001,20000) / 10000
            x2_rand  = np.random.randint(5001,10001,20000) / 10000
            x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand})

    elif var2 == "K":
        df['Xk'] = np.random.randint(1,    5001, 10000) / 10000
        df_append       = create_dataset(pearson_r=pearson_r*(depth_faktor**2), size=10000, depth=1)
        df_append['Xt'] = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df_append['Xk'] = np.random.randint(5001,10001, 10000) / 10000
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X":
            df['Xi'] /= 2
            x1_rand = np.random.randint(5001,  10001, 20000) / 10000
            x2_rand = np.random.randint(1,     10001, 20000) / 10000
            x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            x4_rand = np.random.randint(1,     10001, 20000) / 10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand, 'Xk': x4_rand})
        elif var1 == "Y":
            df['Y'] /= 2
            x1_rand = np.random.randint(1,    10001, 20000) / 10000
            x2_rand = np.random.randint(5001, 10001, 20000) / 10000
            x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            x4_rand = np.random.randint(1,    10001, 20000) / 10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand, 'Xk': x4_rand})

    elif var2 == "Y": #var3 = T   | CAVE: Hier neuer Datensatz, um die Limits für den k-Split an die Steigung des späteren y-Splits anzupassen
                                        #Außerdem: Indextausch zwischen x- und y-Achse
        df = create_dataset(pearson_r=pearson_r, size=1000, depth=3, xlim_min=0, xlim_max = 0.5)
        x3 = ["A"] * 1000
        df['Xt'] = x3
        df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size=9000, depth=2, xlim_max = 0.5)
        x3_append = np.random.choice(["B", "C", "D", "E"])
        df_append['Xt'] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        df_append       = create_dataset(pearson_r=pearson_r*(depth_faktor**2), size=10000, depth=1, xlim_min=0.5, xlim_max = 1)
        df_append['Xt'] = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df = pd.concat([df, df_append], ignore_index=True)
        Xi, Y = df['Y'], df['Xi']                                               #x- und y-Achse werden getauscht!
        df['Y'], df['Xi'] = Y, Xi

        if var1 == "X": #var2 = y, var3 = T
            df['Xi'] /= 2
            x1_rand = np.random.randint(5001, 10001, 20000) / 10000
            x2_rand = np.random.randint(1,    10001, 20000) / 10000
            x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand})

        elif var1 == "K":
            df['Xk'] = np.random.randint(1,   5001, 20000) / 10000
            x1_rand  = np.random.randint(1,  10001, 20000) / 10000
            x2_rand  = np.random.randint(1,  10001, 20000) / 10000
            x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            x4_rand = np.random.randint(5001,10001, 20000) / 10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand, 'Xk': x4_rand})


    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)

#create_var1_var2_T(1, "K", "X", printit = True)
#create_var1_var2_T(1, "Y", "X", printit = True)
#create_var1_var2_T(1, "X", "K", printit = True)
#create_var1_var2_T(1, "Y", "K", printit = True)
#create_var1_var2_T(1, "X", "Y", printit = True)
#create_var1_var2_T(1, "K", "Y", printit = True)

def create_var1_var2_Y(pearson_r, var1, var2, printit = False):
    df = create_dataset(pearson_r=pearson_r, size = 1000, xlim_min = 0, xlim_max = 0.1, depth = 3)
    Xi, Y = df["Y"], df["Xi"]
    df["Xi"], df["Y"] = Xi, Y
    df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size = 9000, xlim_min = 0.1, xlim_max = 1, depth = 2)
    Xi, Y = df_append["Y"], df_append["Xi"]
    df_append["Xi"], df_append["Y"] = Xi, Y
    df = pd.concat([df, df_append], ignore_index= True)

    if var2 == "X": #var3 = Y   | Cave! (noch keine ideale Lösung gefunden. Die vorliegende Lösung ist jedoch zweckmäßig
                    #             ... und weist in jedem Schritt eine Zunahme an Korrelation und einen p-Wert < 0.05 auf
        df["Xi"] /= 2
        df_append = create_dataset(pearson_r=pearson_r * (depth_faktor**2), size = 10000, depth = 1)
        Xi, Y = df_append["Y"], df_append["Xi"]
        df_append["Xi"], df_append["Y"] = Xi/2 + 0.5, Y
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "K": #var2 = X, var3 = Y
            df['Xk']= np.random.randint(1,     5001, 20000) / 10000
            x1_rand = np.random.randint(1,    10001, 20000) / 10000
            x2_rand = np.random.randint(1,    10001, 20000) / 10000
            x3_rand = np.random.randint(5001, 10001, 20000) / 10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})

        elif var1 == "T": #var2 = X, var3 = Y
            df['Xt']= ["C"] * 20000
            x1_rand = np.random.randint(1,    10001, 20000) / 10000
            x2_rand = np.random.randint(1,    10001, 20000) / 10000
            x3_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand})

    elif var2 == "K":
        df["Xk"]       = np.random.randint(1,     5001, 10000) / 10000
        df_append      = create_dataset(pearson_r=pearson_r * (depth_faktor**2), size = 10000, depth = 1)
        df_append["Xk"]= np.random.randint(5001, 10001, 10000) / 10000
        Xi, Y = df_append["Y"], df_append["Xi"]
        df_append["Xi"], df_append["Y"] = Xi, Y
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = K, var3 = Y
            df['Xi'] /= 2
            x1_rand = np.random.randint(5001, 10001, 20000) / 10000
            x2_rand = np.random.randint(1,    10001, 20000) / 10000
            x3_rand = np.random.randint(1,    10001, 20000) / 10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand})

        if var1 == "T": #var2 = K, var3 = Y
            df['Xt'] = ["C"] * 20000
            x1_rand = np.random.randint(1,    10001, 20000) / 10000
            x2_rand = np.random.randint(1,    10001, 20000) / 10000
            x3_rand = np.random.randint(1,    10001, 20000) / 10000
            x4_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xk': x3_rand, 'Xt': x4_rand})

    elif var2 == "T": #var3 = Y
        df["Xt"]       = ["B"] * 10000
        df_append      = create_dataset(pearson_r=pearson_r * (depth_faktor**2), size = 10000, depth = 1)
        df_append["Xt"]= np.random.choice(["A", "C", "D", "E"], 10000)
        Xi, Y = df_append["Y"], df_append["Xi"]
        df_append["Xi"], df_append["Y"] = Xi, Y
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = T, var3 = Y
            df['Xi'] /= 2
            x1_rand = np.random.randint(5001, 10001, 20000) / 10000
            x2_rand = np.random.randint(1,    10001, 20000) / 10000
            x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand})
            
        elif var1 == "K": #var2 = T, var3 = Y
            df['Xk']= np.random.randint(1,     5001, 20000) / 10000
            x1_rand = np.random.randint(1,    10001, 20000) / 10000
            x2_rand = np.random.randint(1,    10001, 20000) / 10000
            x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            x4_rand = np.random.randint(5001, 10001, 20000) / 10000
            df_append = pd.DataFrame({'Xi': x1_rand, 'Y': x2_rand, 'Xt': x3_rand, 'Xk': x4_rand})

    df = pd.concat([df, df_append], ignore_index= True)
    print(stats.pearsonr(df["Xi"], df["Y"]))
    print(stats.pearsonr(df["Xi"][df["Xk"] < 0.5], df["Y"][df["Xk"] < 0.5]))
    print(stats.pearsonr(df["Xi"][df["Y"] < 0.5], df["Y"][df["Y"] < 0.5]))
    if printit: print_dataset(df)
    return df

#create_var1_var2_Y(1, "K", "X", printit = True)
#create_var1_var2_Y(1, "T", "X", printit = True)
#create_var1_var2_Y(1, "X", "K", printit = True)
#create_var1_var2_Y(1, "T", "K", printit = True)
#create_var1_var2_Y(1, "X", "T", printit = True)
#create_var1_var2_Y(1, "K", "T", printit = True)


#____________________________________________________________
#____________________________________________________________