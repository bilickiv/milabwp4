import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
pd.set_option('display.width', 300)
np.set_printoptions(linewidth=300)
pd.set_option('display.max_columns',300)



def filter_useless_cols():

    """
    filters the useless columns and creates a correlation matrix
    :return: None
    """
    #df1 = pd.read_csv(r"I:\Munka\Elso\20230321adatok\alap\szte_skizo.csv")
    df = pd.read_csv(r"features.csv")


    print(df.shape[1])
    df = df.rename(columns={"category": "class"})
    y = df['class'].loc[df['class'] != 3].values
    X = df.loc[df['class'] != 3].drop(['id','class'], axis=1)
    mutual_info = mutual_info_classif(X, y, random_state=42)
    entropy = X.apply(lambda x: -sum(x * np.log2(x)), axis=0)
    scores3 = pd.DataFrame({'Feature': X.columns, 'Mutual_Information': mutual_info, 'Entropy': entropy})
    scores3 = scores3.sort_values(by='Mutual_Information', ascending=False)
    scores3 = scores3[scores3.Mutual_Information == 0]
    cols_to_drop = scores3.Feature.values.tolist()
    df.drop(cols_to_drop, axis=1, inplace=True)
    cols = df.columns
    for co in cols:
        if "_thrd" in co:
            df.drop([co], axis=1, inplace=True)
    print(df.shape[1])

    columns = df.columns
    corr_matrix = abs(df[columns].corr())
    print(corr_matrix)

    corr_matrix.to_csv(r"corr_matrix.csv")

filter_useless_cols()