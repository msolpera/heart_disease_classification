from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


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
    
"""
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

"""
class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, impute_cols, invalid_conditions):
        """
        Replaces anomalous or missing values in specified columns using the overall column median.

        Args:
            impute_cols (list): List of column names to apply imputation on.
            invalid_conditions (dict): Dictionary where keys are column names and values are
                                       functions that return a boolean mask for invalid entries.
                                       Example: {'Cholesterol': lambda x: pd.isna(x) | (x == 0)}
        """
        self.impute_cols = impute_cols
        self.invalid_conditions = invalid_conditions
        self.medians_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.medians_ = {}
        for col in self.impute_cols:
            valid_mask = ~self.invalid_conditions[col](X[col])
            self.medians_[col] = X.loc[valid_mask, col].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.impute_cols:
            invalid_mask = self.invalid_conditions[col](X[col])
            X.loc[invalid_mask, col] = self.medians_[col]
        return X


def preprocessor(cat_features, num_features, fit=False, X=None, y=None):
    """
    Builds a preprocessing pipeline for categorical and numerical features.

    This function creates a scikit-learn pipeline that imputes missing values,
    encodes categorical variables using one-hot encoding, and scales numerical
    variables using MinMaxScaler. Optionally, the pipeline can be fitted
    immediately on the provided data.

    Args:
        cat_features (list): List of names or indices of categorical features.
        num_features (list): List of names or indices of numerical features.
        fit (bool, optional): If True, fits the pipeline on X and y. Default is False.
        X (DataFrame or array-like, optional): Feature data to fit the pipeline on.
        y (Series or array-like, optional): Target variable (not used but passed for compatibility).

    Returns:
        sklearn.pipeline.Pipeline: A pipeline that applies preprocessing to the input data.
    """
    cat_prepro = Pipeline(
        [
            ("imputation_none", SimpleImputer(missing_values=np.nan,
                                              strategy='constant',
             fill_value='NA', add_indicator=True)),
             ("one_hot", OneHotEncoder(sparse_output=False))
        ],
        verbose=False)

    num_prepro = Pipeline(
        [
            ("imputation_none", SimpleImputer(missing_values=np.nan, strategy="median", add_indicator=True)),
            ("scaler", MinMaxScaler())
        ],
        verbose=False
    )


    feature_eng = ColumnTransformer(
        [
            ("cat", cat_prepro, cat_features),
            ("num", num_prepro, num_features),
        ],
        remainder="passthrough",
        verbose=False,
        verbose_feature_names_out=False,
    )  

    preprocessor = Pipeline([("feature_eng", feature_eng)])

    if fit and X is not None and y is not None:
        preprocessor.fit(X, y)

    return preprocessor


