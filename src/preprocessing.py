import pandas as pd
import numpy as np

def drop_nulls(df):
    """
    Loads and cleans the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned dataset 
    """

    # Drop NaN rows
    df_clean = df.dropna(how='all').copy()

    return df_clean

def count_anomalous(df):
    """
    """
    chol_zero = len(df[df['Cholesterol']==0])
    N = len(df)
    print(f"\n Number of null values ​​in the Cholesterol feature: {chol_zero}, ({np.round((chol_zero/N)*100,2)}%)")

    op_neg = len(df[df['Oldpeak']<0])
    print(f"\n Number of negative values ​​in the Oldpeak characteristic: {op_neg}, ({np.round((op_neg/N)*100,2)}%)")

    rest_bp = len(df[df["RestingBP"]==0])
    print(f"\n Number of null values ​​in the RestingBP feature: {rest_bp}, ({np.round((rest_bp/N)*100,2)}%)")
    



def imputation(df):
    # Crear copia para no sobreescribir el DataFrame original
    df_imputed = df.copy()
    
    # Values with Oldpeak < 0 replaced by the median of each group
    median_0 = df_imputed[df_imputed['HeartDisease'] == 0]['Oldpeak'].median()
    median_1 = df_imputed[df_imputed['HeartDisease'] == 1]['Oldpeak'].median()

    df_imputed.loc[(df_imputed['HeartDisease'] == 0) & (df_imputed['Oldpeak'] < 0), 'Oldpeak'] = median_0
    df_imputed.loc[(df_imputed['HeartDisease'] == 1) & (df_imputed['Oldpeak'] < 0), 'Oldpeak'] = median_1

    # Values with RestingBP == 0 replaced by the median of each group
    median_0 = df_imputed[df_imputed['HeartDisease'] == 0]['RestingBP'].median()
    median_1 = df_imputed[df_imputed['HeartDisease'] == 1]['RestingBP'].median()

    df_imputed.loc[(df_imputed['HeartDisease'] == 0) & (df_imputed['RestingBP'] == 0), 'RestingBP'] = median_0
    df_imputed.loc[(df_imputed['HeartDisease'] == 1) & (df_imputed['RestingBP'] == 0), 'RestingBP'] = median_1

    # Cholesterol: Impute 0 (erroneous) values with values distributed by cardiac risk group
    df_imputed['Cholesterol'] = df_imputed['Cholesterol'].replace(0, np.nan)

    def impute_by_group(group_data):
        if group_data.isna().sum() > 0:
            valid_values = group_data.dropna()
            # Crear una Serie con los valores imputados
            imputed_values = pd.Series(
                np.random.choice(valid_values, size=group_data.isna().sum()),
                index=group_data[group_data.isna()].index
            )
            # Llenar los valores faltantes
            group_data = group_data.copy()
            group_data.loc[group_data.isna()] = imputed_values
        return group_data

    df_imputed['Cholesterol'] = df_imputed.groupby('HeartDisease')['Cholesterol'].transform(impute_by_group)

    return df_imputed
