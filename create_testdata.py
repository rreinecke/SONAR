import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


# A designated split allways increases the correlation between the variable "X" and "Y".
# Both "X" and "Y" have values in [0,1].
# The split can happen in both "X" and "Y" as well as in a third or fourth Variable "Kn" or "Kt".
# "Kn" is a continuous variable with values in [0,1].
# "Kt" is categorical with values in {"A", "B", "C", "D", "E"}.


seed_val = None
# Zur Generierung pseudozufälliger, reproduzierbarer Ergebnisse
def SEED():
    global seed_val
    np.random.seed(seed_val)
    seed_val = seed_val + 1 if seed_val is not None else seed_val



depth_faktor = 2/3
type_list = ["linear", "square", "cub", "exp", "exp_n", "sqrt"]


def print_dataset(df):
    df_plt_obj = sns.pairplot(df, diag_kind="kde", plot_kws=dict(marker="x", alpha = 0.9, linewidth=0.08))
    #df_plt_obj.map_lower(sns.kdeplot, levels=4, color="0.2")
    plt.show()


#____________________________________________________________
#____________________________________________________________


def create_dataset(pearson_r, size=1000, xlim_min = 0, xlim_max =1, depth = 1, type = "linear"):
    m = pearson_r*0.6* ((depth_faktor)**(depth-1))
    n = 0.5-0.5*m
    alpha = 0.001
    x1 = np.arange(1, size + 1)
    p_value, corr, exit_p = 1, -999, 0
    while True:
        exit_p = exit_p + 1
        SEED(); x2_independent = np.random.normal(0, 1, size)
        x2 = (pearson_r * (x1 - np.mean(x1)) / np.std(x1) + np.sqrt(1 - pearson_r ** 2) * x2_independent)
        a, b = np.polyfit(x1, x2, 1)   #identifizieren der Regressionsgeraden y = a*x+b
        x2 = ((x2 - b) / a) / size           # normieren auf y = 1*x + 0
        x2 = x2 *m*(xlim_max-xlim_min)      #Anpassen des Steigungsfaktors an die gewünschte Form
        x2 = x2 + n + m*xlim_min            #Anpassen des y-Achsen Abschnitts an die gewünschte Form
        for i in range(len(x2)):
            if (x2[i] <= 0) or (x2[i] >1):
                SEED(); x2[i] = np.random.randint(1, 10001) / 10000

        corr, p_value = stats.pearsonr(x1, x2)
        # Checkt, ob alle Bedingungen erfüllt sind.
        if (p_value <= alpha) & (abs(corr - pearson_r) < 0.01):
            break
        # Abbruchbedingung, falls kein Ergebnis erreicht wird.
        if exit_p >= 1000:
            print("Requirements not satisfied")
            return

    if type == "linear":
        x2 = x2
    elif type == "square":
        x2 = x2**2
    elif type == "cub":
        x2 = x2**3
    elif type == "exp":
        x2 = (10**x2) / 10
    elif type == "exp_n":
        x2 = (10**(-x2))
    elif type == "sqrt":
        x2 = np.sqrt(x2)
    else:
        print(f"Type '{type}' does not exist.")
        exit()

    # Erzeugt ein Pandas DF
    df = pd.DataFrame({'X': (x1*(xlim_max-xlim_min)/size) + xlim_min, 'Y':x2})
    return df


#____________________________________________________________
#____________________________________________________________


def create_X(pearson_r, printit = False, type = "linear"):
    df = create_dataset(pearson_r = pearson_r, type = type)
    df["X"] = df["X"]/10
    SEED(); x1_rand = np.random.randint(1001,10001, 9000)/10000
    SEED(); x2_rand = np.random.randint(1,10001, 9000)/10000
    df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand})
    df = pd.concat([df, df_append], ignore_index= True)
    if printit: print_dataset(df)
    return df

#create_X(1, printit = True, type = "linear")

def create_Kn(pearson_r, printit = False, type = "linear"):
    df = create_dataset(pearson_r=pearson_r, type = type)

    SEED(); x3_df = np.random.randint(1,1001, 1000)/10000
    df["Kn"] =x3_df

    SEED(); x1_rand = np.random.randint(1,10001, 9000)/10000
    SEED(); x2_rand = np.random.randint(1,10001, 9000)/10000
    SEED(); x3_rand = np.random.randint(1001,10001, 9000)/10000
    df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, "Kn": x3_rand})
    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#df = create_Kn(0.99, printit = True, type = "linear")

def create_Kt(pearson_r, printit = False, type = "linear"):
    df = create_dataset(pearson_r=pearson_r, type = type)
    x3_df =  ["A"] * 1000
    df["Kt"] = x3_df
    SEED(); x1_rand = np.random.randint(1,10001, 9000)/10000
    SEED(); x2_rand = np.random.randint(1,10001, 9000)/10000
    SEED(); x3_rand = np.random.choice(["B", "C", "D", "E"], 9000)
    df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, "Kt": x3_rand})
    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#df = create_Kt(0.999, printit = True, type = "linear")

def create_Y(pearson_r, printit = False, type = "linear"):
    df = create_dataset(pearson_r=pearson_r, type = type)
    X = df["Y"]
    Y  = df["X"] / 10
    df = pd.DataFrame({'X': X, 'Y': Y})
    SEED(); x1_rand = np.random.randint(1,10001, 9000)/10000
    SEED(); x2_rand = np.random.randint(1001,10001, 9000)/10000
    df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand})
    df = pd.concat([df, df_append], ignore_index= True)
    if printit: print_dataset(df)
    return df

#df = create_Y(0.99, printit = True, type = "linear")


#____________________________________________________________
#____________________________________________________________


def create_var1_X(pearson_r, var1, printit = False, type = "linear"):
    df_sign = create_dataset(pearson_r=pearson_r,              size=1000, xlim_min = 0, xlim_max = 0.1, depth = 2, type = type)
    df_rest = create_dataset(pearson_r=pearson_r*depth_faktor, size=9000, xlim_min = 0.1, xlim_max = 1, depth = 1, type = type)
    df = pd.concat([df_sign, df_rest], ignore_index=True)

    if var1 == "Kn":
        SEED(); x3 = np.random.randint(1,5001, 10000)/10000
        df["Kn"] = x3
        SEED(); x1_rand = np.random.randint(1,    10001, 10000)/10000
        SEED(); x2_rand = np.random.randint(1,    10001, 10000)/10000
        SEED(); x3_rand = np.random.randint(5001, 10001, 10000)/10000
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})

    elif var1 == "Kt":
        x3 =  ["B"] * 10000
        df["Kt"] = x3
        SEED(); x1_rand = np.random.randint(1, 10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(1, 10001, 10000) / 10000
        SEED(); x3_rand = np.random.choice(["A", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand})

    elif var1 == "Y":
        df["Y"] = df["Y"]/2
        SEED(); x1_rand = np.random.randint(1,   10001, 10000)/10000
        SEED(); x2_rand = np.random.randint(5001,10001, 10000)/10000
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand})

    df = pd.concat([df, df_append], ignore_index=True)
    if printit:
            print_dataset(df)

    return df

#create_var1_X(1, "Kn", printit = True, type = "linear")
#create_var1_X(1, "Kt", printit = True, type = "linear")
#create_var1_X(0.8, "Y", printit = True, type = "linear")

def create_var1_Kn(pearson_r, var1, printit = False, type = "linear"):
    df = create_dataset(pearson_r=pearson_r, size = 1000, depth = 2, type = type)
    SEED(); x3_df = np.random.randint(1,1001, 1000)/10000
    df["Kn"] =x3_df
    df_append = create_dataset(pearson_r=pearson_r*depth_faktor, size = 9000, depth = 1, type = type)
    SEED(); x3_append = np.random.randint(1001, 10001, 9000)/10000
    df_append["Kn"] =x3_append
    df = pd.concat([df, df_append], ignore_index=True)

    if var1 == "X":
        df["X"] /=2
        SEED(); x1_rand = np.random.randint(5001,10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x3_rand = np.random.randint(1,   10001, 10000) / 10000
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})
    elif var1 == "Kt":
        x4 = ["B"] * 10000
        df["Kt"] = x4
        SEED(); x1_rand = np.random.randint(1,10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(1,10001, 10000) / 10000
        SEED(); x3_rand = np.random.randint(1,10001, 10000) / 10000
        SEED(); x4_rand = np.random.choice(["A", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand, 'Kt': x4_rand})
    elif var1 == "Y":
        df["Y"] /= 2
        SEED(); x1_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(5001,10001, 10000) / 10000
        SEED(); x3_rand = np.random.randint(1,   10001, 10000) / 10000
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})
    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_Kn(1,"X", printit = True, type = "linear")
#create_var1_Kn(1,"Kt", printit = True, type = "linear")
#create_var1_Kn(1,"Y", printit = True, type = "linear")

def create_var1_Kt(pearson_r, var1, printit = False, type = "linear"):
    df = create_dataset(pearson_r=pearson_r, size = 1000, depth = 2, type = type)
    x3 = ["A"] * 1000
    df['Kt'] = x3
    df_append = create_dataset(pearson_r=pearson_r*depth_faktor, size = 9000, depth = 1, type = type)
    SEED(); x3_append = np.random.choice(["B", "C", "D", "E"], 9000)
    df_append['Kt'] = x3_append
    df = pd.concat([df, df_append], ignore_index=True)
    if var1 == "X":
        df['X'] /= 2
        SEED(); x1_rand = np.random.randint(5001,10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand})
    if var1 == "Kn":
        SEED(); x4_append = np.random.randint(1,    5001, 10000) / 10000
        df['Kn'] = x4_append
        SEED(); x1_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        SEED(); x4_rand = np.random.randint(5001,10001, 10000) / 10000
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand, 'Kn':x4_rand})
    if var1 == "Y":
        df['Y'] /= 2
        SEED(); x1_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(5001,10001, 10000) / 10000
        SEED(); x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand})
    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_Kt(.1,"X",True, type = "linear")
#create_var1_Kt(1,"Kn",True, type = "linear")
#create_var1_Kt(1,"Y",True, type = "linear")

def create_var1_Y(pearson_r, var1, printit = False, type = "linear"):
    df = create_dataset(pearson_r=pearson_r, size = 1000, xlim_min = 0, xlim_max = 0.1, depth = 2, type = type)
    X, Y = df["Y"], df["X"]
    df["X"], df["Y"] = X, Y
    df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size = 9000, xlim_min = 0.1, xlim_max = 1, depth = 1, type = type)
    X, Y = df_append["Y"], df_append["X"]
    df_append["X"], df_append["Y"] = X, Y
    df = pd.concat([df, df_append], ignore_index= True)
    if var1 == "X":
        df["X"] /= 2
        SEED(); x1_rand = np.random.randint(5001,10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(1   ,10001, 10000) / 10000
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand})
    if var1 == "Kn":
        SEED(); x3_append = np.random.randint(1,5001, 10000) / 10000
        df["Kn"] = x3_append
        SEED(); x1_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x3_rand = np.random.randint(5001,10001, 10000) / 10000
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn':x3_rand})
    if var1 == "Kt":
        x3_append = ["B"] * 10000
        df['Kt'] = x3_append
        SEED(); x1_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x2_rand = np.random.randint(1,   10001, 10000) / 10000
        SEED(); x3_rand = np.random.choice(["A", "C", "D", "E"], 10000)
        df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt':x3_rand})
    df = pd.concat([df, df_append], ignore_index= True)
    if printit: print_dataset(df)
    return df

#create_var1_Y(1, "X", printit = True, type = "linear")
#create_var1_Y(1, "Kn", printit = True, type = "linear")
#create_var1_Y(1, "Kt", printit = True, type = "linear")

#____________________________________________________________
#____________________________________________________________


def create_var1_var2_X(pearson_r, var1, var2, printit = False, type = "linear"):
    df_sign = create_dataset(pearson_r=pearson_r,             size=1000, xlim_min = 0, xlim_max = 0.1, depth = 3, type = type)
    df_rest = create_dataset(pearson_r=pearson_r*depth_faktor,size=9000, xlim_min = 0.1, xlim_max = 1, depth = 2, type = type)
    df = pd.concat([df_sign, df_rest], ignore_index=True)

    if var2 == "Kt": #var3 = X
        x3 =  ["B"] * 10000
        df["Kt"] = x3
        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2),size=10000, xlim_min = 0, xlim_max = 1, depth = 1, type = type)
        SEED(); x3_append = np.random.choice(["A", "C", "D", "E"], 10000)
        df_append['Kt'] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "Kn": #var2 = Kt, var3 = X
            SEED(); x4 = np.random.randint(1,   5001, 20000) / 10000
            df['Kn'] = x4
            SEED(); x1_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            SEED(); x4_rand = np.random.randint(5001, 10001, 20000) / 10000
            df_append = pd.DataFrame({'X':x1_rand, 'Y': x2_rand, 'Kt': x3_rand, 'Kn': x4_rand})

        elif var1 == "Y": #var2 = Kt, var3 = X
            df["Y"] /= 2
            SEED(); x1_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(5001,10001, 20000) / 10000
            SEED(); x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'X':x1_rand, 'Y': x2_rand, 'Kt': x3_rand})

    elif var2 == "Kn": #var3 = X
        SEED(); x3 = np.random.randint(1,5001, 10000)/10000
        df["Kn"] = x3
        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2),size=10000, xlim_min = 0, xlim_max = 1, depth = 1, type = type)
        SEED(); x3_append = np.random.randint(5001, 10001, 10000)/10000
        df_append['Kn'] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "Kt": #var2 = Kn, var3 = X
            x4 = ["C"]*20000
            df['Kt'] = x4
            SEED(); x1_rand = np.random.randint(1,10001, 20000)/10000
            SEED(); x2_rand = np.random.randint(1,10001, 20000)/10000
            SEED(); x3_rand = np.random.randint(1,10001, 20000)/10000
            SEED(); x4_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'X':x1_rand, 'Y': x2_rand, 'Kn': x3_rand, 'Kt': x4_rand})

        elif var1 == "Y": #var2 = Kn, var3 = X
            df["Y"] /= 2
            SEED(); x1_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x2_rand = np.random.randint(5001,10001, 20000)/10000
            SEED(); x3_rand = np.random.randint(1,   10001, 20000)/10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})

    elif var2 == "Y": #var3 = X
        df["Y"]  = df["Y"]/2
        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2),size=10000, xlim_min = 0, xlim_max = 1, depth = 1, type = type)
        df_append['Y'] = df_append['Y']/2 + 0.5
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "Kn": #var2 = Y, var3 = X
            SEED(); x3 = np.random.randint(1,5001, 20000) / 10000
            df['Kn'] = x3
            SEED(); x1_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x2_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x3_rand = np.random.randint(5001,10001, 20000)/10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})

        if var1 == "Kt": #var2 = Y, var3 = X
            x3 = ["C"] * 20000
            df['Kt'] = x3
            SEED(); x1_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x2_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x3_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand})

    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_var2_X(1,"Kn", "Kt", printit = True, type = "linear")
#create_var1_var2_X(1,"Y",  "Kt", printit = True, type = "linear")
#create_var1_var2_X(1,"Kt", "Kn", printit = True, type = "linear")
#create_var1_var2_X(1,"Y",  "Kn", printit = True, type = "linear")
#create_var1_var2_X(1,"Kn", "Y", printit = True, type = "linear") #Cave; eigentlich nicht!
#create_var1_var2_X(1,"Kt", "Y", printit = True, type = "linear") #Cave; eigentlich nicht!

def create_var1_var2_Kn(pearson_r, var1, var2, printit = False, type = "linear"):

    df = create_dataset(pearson_r=pearson_r, size = 1000, depth = 3, type = type)
    SEED(); x3_df = np.random.randint(1,1001, 1000)/10000
    df["Kn"] =x3_df
    df_append = create_dataset(pearson_r=pearson_r*depth_faktor, size = 9000, depth = 2, type = type)
    SEED(); x3_append = np.random.randint(1001, 10001, 9000)/10000
    df_append["Kn"] =x3_append
    df = pd.concat([df, df_append], ignore_index=True)


    if var2 == "X": #var3 = Kn     CAVE: Hier neuer Datensatz, um die Limits für den x-Split an die Steigung des späteren k-Splits anzupassen
        df = create_dataset(pearson_r=pearson_r, size=1000, depth=3, xlim_min = 0, xlim_max = 0.5, type = type)
        SEED(); x3_df = np.random.randint(1, 1001, 1000) / 10000
        df["Kn"] = x3_df
        df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size=9000, depth=2, xlim_min = 0, xlim_max = 0.5, type = type)
        SEED(); x3_append = np.random.randint(1001, 10001, 9000) / 10000
        df_append["Kn"] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        #Datensatz für den X-Split
        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2),size=10000, xlim_min = 0.5, xlim_max = 1, depth = 1, type = type)
        SEED(); x3_rand = np.random.randint(1,   10001, 10000) / 10000
        df_append["Kn"] = x3_rand
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "Kt": #var2 = X var3 = Kn
            df["Kt"] =  ["C"] * 20000
            SEED(); x1_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x2_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x3_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x4_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'X':x1_rand, 'Y':x2_rand, 'Kn': x3_rand, 'Kt':x4_rand})

        if var1 == "Y": #var2 = X var3 = Kn
            df["Y"] /= 2
            SEED(); x1_rand = np.random.randint(1,   10001, 20000)/10000
            SEED(); x2_rand = np.random.randint(5001,10001, 20000)/10000
            SEED(); x3_rand = np.random.randint(1,   10001, 20000)/10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})

    elif var2 == "Kt": #var3 = Kn
        x4 = ["B"] * 10000
        df["Kt"] = x4
        df_append = create_dataset(pearson_r = pearson_r*(depth_faktor**2), size = 10000, xlim_min = 0, xlim_max = 1, depth = 1, type = type)
        SEED(); df_append['Kn'] = np.random.randint(1,10001, 10000) / 10000
        SEED(); df_append['Kt'] = np.random.choice(["A", "C", "D", "E"], 10000)
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = Kt, var3 = Kn
            df["X"] /= 2
            SEED(); x1_rand = np.random.randint(5001,10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x3_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x4_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand, 'Kt': x4_rand})

        if var1 == "Y": #var2 = Kt, var3 = Kn
            df["Y"] /= 2
            SEED(); x1_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(5001,10001, 20000) / 10000
            SEED(); x3_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x4_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand, 'Kt': x4_rand})

    elif var2 == "Y": #var3 = Kn     | || CAVE: Hier neuer Datensatz, um die Limits für den x-Split an die Steigung des späteren k-Splits anzupassen
                      #                        Außerdem Index-Shift (X und Y vertauscht, um nach vor dem Split in Y noch einen sinnvollen Split einzubauen.
        #Neuer Datensatz (Eigentlich wie der "alte" nur mit veränderten Limits aufgrund der Steigung)
        df = create_dataset(pearson_r=pearson_r, size=1000, depth=3, xlim_min = 0, xlim_max = 0.5, type = type)
        SEED(); x3_df = np.random.randint(1, 1001, 1000) / 10000
        df["Kn"] = x3_df
        df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size=9000, depth=2, xlim_min = 0, xlim_max = 0.5, type = type)
        SEED(); x3_append = np.random.randint(1001, 10001, 9000) / 10000
        df_append["Kn"] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        #Index-Shift
        X, Y = df["Y"], df["X"]
        df["X"], df["Y"] = X, Y
        df_append = create_dataset(pearson_r = pearson_r*(depth_faktor**2), size = 10000, xlim_min = 0.5, xlim_max = 1, depth = 1, type = type)
        X, Y = df_append["Y"], df_append["X"]
        df_append["X"], df_append["Y"] = X, Y
        SEED(); df_append["Kn"] = np.random.randint(1,   10001, 10000) / 10000
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = Y, var3 = Kn
            df['X'] /= 2
            SEED(); x1_rand = np.random.randint(5001,10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x3_rand = np.random.randint(1,   10001, 20000) / 10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})

        elif var1 == "Kt": #var2 = Y, var3 = Kn
            df["Kt"] = ["C"] * 20000
            SEED(); x1_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x3_rand = np.random.randint(1,   10001, 20000) / 10000
            SEED(); x4_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand, 'Kt': x4_rand})

    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_var2_Kn(pearson_r = 1, var1 = "Kt", var2 = "X",  printit = True, type = "linear")
#create_var1_var2_Kn(pearson_r = 1, var1 = "Y",  var2 = "X",  printit = True, type = "linear")
#create_var1_var2_Kn(pearson_r = 1, var1 = "X",  var2 = "Kt", printit = True, type = "linear")
#create_var1_var2_Kn(pearson_r = 1, var1 = "Y",  var2 = "Kt", printit = True, type = "linear")
#create_var1_var2_Kn(pearson_r = 1, var1 = "X",  var2 = "Y",  printit = True, type = "linear")
#create_var1_var2_Kn(pearson_r = 1, var1 = "Kt", var2 = "Y",  printit = True, type = "linear")

def create_var1_var2_Kt(pearson_r, var1, var2, printit = False, type = "linear"):
    df = create_dataset(pearson_r=pearson_r, size = 1000, depth = 3, xlim_min = 0, type = type)
    x3 = ["A"] * 1000
    df['Kt'] = x3
    df_append = create_dataset(pearson_r=pearson_r*depth_faktor, size = 9000, depth = 2, type = type)
    SEED(); x3_append = np.random.choice(["B", "C", "D", "E"], 9000)
    df_append['Kt'] = x3_append
    df = pd.concat([df, df_append], ignore_index=True)

    if var2 == "X": #var3 = Kt     CAVE: Hier neuer Datensatz, um die Limits für den x-Split an die Steigung des späteren t-Splits anzupassen
        df = create_dataset(pearson_r=pearson_r, size=1000, xlim_min = 0, xlim_max = 0.5, depth=3, type = type)
        x3 = ["A"] * 1000
        df['Kt'] = x3
        df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size=9000, xlim_max = 0.5, depth = 2, type = type)
        SEED(); x3_append = np.random.choice(["B", "C", "D", "E"], 9000)
        df_append['Kt'] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        df_append = create_dataset(pearson_r=pearson_r*(depth_faktor**2), size=10000, xlim_min = 0.5, xlim_max = 1, depth=1, type = type)
        SEED(); df_append['Kt']  = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "Kn": #var2 = X, var3 = Kt
            SEED(); df["Kn"] = np.random.randint(1,    5001,20000) / 10000
            SEED(); x1_rand= np.random.randint(1,   10001,20000) / 10000
            SEED(); x2_rand= np.random.randint(1,   10001,20000) / 10000
            SEED(); x3_rand= np.random.choice(["A", "B", "C", "D", "E"], 20000)
            SEED(); x4_rand= np.random.randint(5001,10001,20000) / 10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand, 'Kn': x4_rand})

        elif var1 == "Y": #var2 = X, var3 = Kt
            df['Y'] /= 2
            SEED(); x1_rand  = np.random.randint(1,   10001,20000) / 10000
            SEED(); x2_rand  = np.random.randint(5001,10001,20000) / 10000
            SEED(); x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand})

    elif var2 == "Kn": #var3 = Kt
        SEED(); df['Kn']        = np.random.randint(1,    5001, 10000) / 10000
        df_append               = create_dataset(pearson_r=pearson_r*(depth_faktor**2), size=10000, depth=1, type = type)
        SEED(); df_append['Kt'] = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        SEED(); df_append['Kn'] = np.random.randint(5001,10001, 10000) / 10000
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = Kn, var3 = Kt
            df['X'] /= 2
            SEED(); x1_rand = np.random.randint(5001,  10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,     10001, 20000) / 10000
            SEED(); x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            SEED(); x4_rand = np.random.randint(1,     10001, 20000) / 10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand, 'Kn': x4_rand})
        elif var1 == "Y": #var2 = Kn, var3 = Kt
            df['Y'] /= 2
            SEED(); x1_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(5001, 10001, 20000) / 10000
            SEED(); x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            SEED(); x4_rand = np.random.randint(1,    10001, 20000) / 10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand, 'Kn': x4_rand})

    elif var2 == "Y": #var3 = Kt   | CAVE: Hier neuer Datensatz, um die Limits für den k-Split an die Steigung des späteren y-Splits anzupassen
                                        #Außerdem: Indextausch zwischen x- und y-Achse
        df = create_dataset(pearson_r=pearson_r, size=1000, depth=3, xlim_min=0, xlim_max = 0.5, type = type)
        x3 = ["A"] * 1000
        df['Kt'] = x3
        df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size=9000, depth=2, xlim_max = 0.5, type = type)
        SEED(); x3_append = np.random.choice(["B", "C", "D", "E"], 9000)
        df_append['Kt'] = x3_append
        df = pd.concat([df, df_append], ignore_index=True)

        df_append       = create_dataset(pearson_r=pearson_r*(depth_faktor**2), size=10000, depth=1, xlim_min=0.5, xlim_max = 1, type = type)
        SEED(); df_append['Kt'] = np.random.choice(["A", "B", "C", "D", "E"], 10000)
        df = pd.concat([df, df_append], ignore_index=True)
        X, Y = df['Y'], df['X']                                               #x- und y-Achse werden getauscht!
        df['Y'], df['X'] = Y, X

        if var1 == "X": #var2 = Y, var3 = Kt
            df['X'] /= 2
            SEED(); x1_rand = np.random.randint(5001, 10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand})

        elif var1 == "Kn": #var2 = Y, var3 = Kt
            SEED(); df['Kn'] = np.random.randint(1,   5001, 20000) / 10000
            SEED(); x1_rand  = np.random.randint(1,  10001, 20000) / 10000
            SEED(); x2_rand  = np.random.randint(1,  10001, 20000) / 10000
            SEED(); x3_rand  = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            SEED(); x4_rand = np.random.randint(5001,10001, 20000) / 10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand, 'Kn': x4_rand})


    df = pd.concat([df, df_append], ignore_index=True)
    if printit: print_dataset(df)
    return df

#create_var1_var2_Kt(1, "Kn", "X",  printit = True, type = "linear")
#create_var1_var2_Kt(1, "Y",  "X",  printit = True, type = "linear")
#create_var1_var2_Kt(1, "X",  "Kn", printit = True, type = "linear")
#create_var1_var2_Kt(1, "Y",  "Kn", printit = True, type = "linear")
#create_var1_var2_Kt(1, "X",  "Y",  printit = True, type = "linear")
#create_var1_var2_Kt(1, "Kn", "Y",  printit = True, type = "linear")

def create_var1_var2_Y(pearson_r, var1, var2, printit = False, type = "linear"):
    df = create_dataset(pearson_r=pearson_r, size = 1000, xlim_min = 0, xlim_max = 0.1, depth = 3, type = type)
    X, Y = df["Y"], df["X"]
    df["X"], df["Y"] = X, Y
    df_append = create_dataset(pearson_r=pearson_r * depth_faktor, size = 9000, xlim_min = 0.1, xlim_max = 1, depth = 2, type = type)
    X, Y = df_append["Y"], df_append["X"]
    df_append["X"], df_append["Y"] = X, Y
    df = pd.concat([df, df_append], ignore_index= True)

    if var2 == "X": #var3 = Y   | Cave! (noch keine ideale Lösung gefunden. Die vorliegende Lösung ist jedoch zweckmäßig
                    #             ... und weist in jedem Schritt eine Zunahme an Korrelation und einen p-Wert < 0.05 auf
        df["X"] /= 2
        df_append = create_dataset(pearson_r=pearson_r * (depth_faktor**2), size = 10000, depth = 1, type = type)
        X, Y = df_append["Y"], df_append["X"]
        df_append["X"], df_append["Y"] = X/2 + 0.5, Y
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "Kn": #var2 = X, var3 = Y
            SEED(); df['Kn']= np.random.randint(1,     5001, 20000) / 10000
            SEED(); x1_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x3_rand = np.random.randint(5001, 10001, 20000) / 10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})

        elif var1 == "Kt": #var2 = X, var3 = Y
            df['Kt']= ["C"] * 20000
            SEED(); x1_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x3_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand})

    elif var2 == "Kn": #var3 = Y
        SEED(); df["Kn"]       = np.random.randint(1,     5001, 10000) / 10000
        df_append      = create_dataset(pearson_r=pearson_r * (depth_faktor**2), size = 10000, depth = 1, type = type)
        SEED(); df_append["Kn"]= np.random.randint(5001, 10001, 10000) / 10000
        X, Y = df_append["Y"], df_append["X"]
        df_append["X"], df_append["Y"] = X, Y
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = Kn, var3 = Y
            df['X'] /= 2
            SEED(); x1_rand = np.random.randint(5001, 10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x3_rand = np.random.randint(1,    10001, 20000) / 10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand})

        if var1 == "Kt": #var2 = Kn, var3 = Y
            df['Kt'] = ["C"] * 20000
            SEED(); x1_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x3_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x4_rand = np.random.choice(["A", "B", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kn': x3_rand, 'Kt': x4_rand})

    elif var2 == "Kt": #var3 = Y
        df["Kt"]       = ["B"] * 10000
        df_append      = create_dataset(pearson_r=pearson_r * (depth_faktor**2), size = 10000, depth = 1, type = type)
        SEED(); df_append["Kt"]= np.random.choice(["A", "C", "D", "E"], 10000)
        X, Y = df_append["Y"], df_append["X"]
        df_append["X"], df_append["Y"] = X, Y
        df = pd.concat([df, df_append], ignore_index=True)

        if var1 == "X": #var2 = Kt, var3 = Y
            df['X'] /= 2
            SEED(); x1_rand = np.random.randint(5001, 10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand})

        elif var1 == "Kn": #var2 = Kt, var3 = Y
            SEED(); df['Kn']= np.random.randint(1,     5001, 20000) / 10000
            SEED(); x1_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x2_rand = np.random.randint(1,    10001, 20000) / 10000
            SEED(); x3_rand = np.random.choice(["A", "B", "C", "D", "E"], 20000)
            SEED(); x4_rand = np.random.randint(5001, 10001, 20000) / 10000
            df_append = pd.DataFrame({'X': x1_rand, 'Y': x2_rand, 'Kt': x3_rand, 'Kn': x4_rand})

    df = pd.concat([df, df_append], ignore_index= True)
    if printit: print_dataset(df)
    return df

#create_var1_var2_Y(1, "Kn", "X",  printit = True, type = "linear") #Cave: Eigentlich nicht!
#create_var1_var2_Y(1, "Kt", "X",  printit = True, type = "linear") #Cave: Eigentlich nicht!
#create_var1_var2_Y(1, "X",  "Kn", printit = True, type = "linear")
#create_var1_var2_Y(1, "Kt", "Kn", printit = True, type = "linear")
#create_var1_var2_Y(1, "X",  "Kt", printit = True, type = "linear")
#create_var1_var2_Y(1, "Kn", "Kt", printit = True, type = "linear")


#____________________________________________________________
#____________________________________________________________


def test_data(split1, split2, split3, pearson_r, printit = False, type = "linear", kat = False, kog = False, kon = False, seed = None):
    #Check the arguments.
    if (split1 not in ["O", "X", "Kn", "Kt", "Y"]) or (split2 not in ["O", "X", "Kn", "Kt", "Y"]) or (split3 not in ["X", "Kn", "Kt", "Y"]):
        print("Wrong input")
        return
    if (split1 == split2) or (split1 == split3) or (split2 == split3):
        if not ((split1 == "O") and (split2 == "O")):
            print("Split variables must be different in pairs.")
            return
    if (pearson_r < -1) or (pearson_r > 1):
        print("Correlation value (Pearson r) must be between -1 and 1.")
        return
    if type not in ["linear", "square", "cub", "exp", "exp_n", "sqrt"]:
        print("Function type must be one of the following: 'linear', 'square', 'cub', 'exp', 'exp_n' or 'sqrt'.")
        return

    #Soll ein Seed erfolgen?
    global seed_val; seed_val = seed

    if (split1 == "O") and (split2 == "O"):
        function_name = f"create_{split3}"
        func = globals()[function_name]
        df = func(pearson_r = pearson_r, type = type)

    elif (split1 == "O") and (split2 != "O"):
        function_name = f"create_var1_{split3}"
        func = globals()[function_name]
        df = func(pearson_r = pearson_r, var1 = split2, type = type)

    elif (split1 != "O") and (split2 != "O"):
        function_name = f"create_var1_var2_{split3}"
        func = globals()[function_name]
        df = func(pearson_r = pearson_r, var1 = split1, var2 = split2, type = type)

    else: #when split1 == "O" and split2 ≠ 0
        print("Wrong input. Split2 can't be 'O', when split1 ≠ O.")
        return

    size = len(df)
    if "Kt" in [split1, split2, split3]:
        df["Kt"] = pd.Categorical(df["Kt"], categories=["A", "B", "C", "D", "E"])
    if kat:
        SEED(); df["kat"] = np.random.choice(["v", "w", "x", "y", "z"], size)
        df["kat"]         = pd.Categorical(df["kat"], categories=["v", "w", "x", "y", "z"])
    if kog:
        SEED(); df["kog"] = np.random.randint(1, 10001, size) / 10000
    if kon:
        SEED(); x_append = np.random.normal(0, 1, size)
        x_append         = (x_append / 6) + 0.5
        for i in range(len(x_append)):
            if (x_append[i] <= 0) or (x_append[i] >1):
                SEED(); x_append[i] =np.random.randint(1,10001)/10000
        df["kon"] = x_append

    if printit:
        print_dataset(df)
    return df

#Testaufruf
#a = test_data("Kt","Kn","X", 1, type = "linear", printit = False, kat = True, kog = True, kon = True, seed = 1)
#b = test_data("Kt","Kn","X", 1, type = "linear", printit = False, kat = True, kog = True, kon = True, seed = 2)