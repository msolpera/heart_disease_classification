import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def check_nulls(df):

    """
    Prints the number of null values per column in a DataFrame.
    Prints how many rows have at least one null value and shows them.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    print(f"\n Amount of nulls per column")
    print(df.isnull().sum())

    null_rows = df[df.isnull().any(axis=1)]
    print(f"\n Amount of rows that have at least a NaN value: {len(null_rows)}")
    print(null_rows)
    

def plot_distributions(df):
    """
    Plots distribution of all features including HeartDisease data.

    Args:
        df (pd.DataFrame): DataFrame that includes those columns.
    
    Returns:
        None: Displays the plots using matplotlib.
    """
    
    # Calcular dimensiones dinámicamente
    n_cols = len(df.columns)
    n_rows = (n_cols + 2) // 3 
    
    plt.figure(figsize=(15, n_rows * 3))
    
    for i, col in enumerate(df.columns, 1):
        plt.subplot(n_rows, 3, i)
        plt.title(f"Distribution of {col}")
        
        # Manejar diferentes tipos de datos
        try:
            sns.histplot(df[col], kde=True)
        except Exception as e:
            plt.text(0.5, 0.5, f'Cannot plot {col}\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()


def string_num_cols(df):
    """
    Separate categorical and numerical columns
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        cat_features, num_features (Python lists): lists of cat/num features names
    """
    # Transform object type to string type
    string_col = df.select_dtypes("object").columns
    df[string_col]=df[string_col].astype("string")

    cat_features=df.select_dtypes("string").columns.to_list()
    num_features=df.columns.to_list()

    for col in cat_features:
        num_features.remove(col)
    return cat_features, num_features


def describe_num_cols(df, num_features):
    """
    Display statistical summary and correlation matrix
    
    Args:
        df (pd.DataFrame): Input DataFrame
        num_features (list): List of numerical column names
    """
    print(f"\nStatistical data of the numerical columns:")
    print("="*50)
    print(df[num_features].describe().T.round(3))
    
    print(f"\nCorrelation Matrix:")
    print("="*50)
    
    # Crear heatmap
    fig = px.imshow(
        df[num_features].corr(),
        title="Correlation Matrix of Numerical Features",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig.show()

def plot_feature_distributions(df, target='HeartDisease', bins=30, cols=3, 
                                include_categorical=True):
    """
    Plots the distribution of all numerical and optional categorical features 
    in subplots, separated by target class.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the target column (default is 'HeartDisease').
        bins (int): Number of histogram bins for numeric features.
        cols (int): Number of columns in the subplot grid.

        include_categorical (bool): If True, also plots categorical feature distributions.
    """
    df_plot = df.copy()

    # Separate numeric and (optionally) categorical features
    numeric_features = df_plot.select_dtypes(include='number').columns.drop(target)
    categorical_features = []
    if include_categorical:
        categorical_features = df_plot.select_dtypes(include='object').columns.tolist()

    # Plot numeric features
    n_num = len(numeric_features)
    rows_num = (n_num + cols - 1) // cols

    fig_num, axes_num = plt.subplots(rows_num, cols, figsize=(cols * 6, rows_num * 4))
    axes_num = axes_num.flatten()

    for i, feature in enumerate(numeric_features):
        sns.histplot(data=df_plot, x=feature, hue=target, bins=bins,
                     kde=True, stat='density', element='step', ax=axes_num[i])
        axes_num[i].set_title(f"Distribution of {feature}")
        axes_num[i].set_ylabel("Density")

    for j in range(i + 1, len(axes_num)):
        fig_num.delaxes(axes_num[j])

    fig_num.suptitle("Numerical Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.show()

    #Plot categorical features (optional)
    if include_categorical and categorical_features:
        n_cat = len(categorical_features)
        rows_cat = (n_cat + cols - 1) // cols

        fig_cat, axes_cat = plt.subplots(rows_cat, cols, figsize=(cols * 6, rows_cat * 4))
        axes_cat = axes_cat.flatten()

        for i, feature in enumerate(categorical_features):
            sns.countplot(data=df_plot, x=feature, hue=target, ax=axes_cat[i])
            axes_cat[i].set_title(f"Distribution of {feature}")
            axes_cat[i].tick_params(axis='x', rotation=30)

        for j in range(i + 1, len(axes_cat)):
            fig_cat.delaxes(axes_cat[j])

        fig_cat.suptitle("Categorical Feature Distributions", fontsize=16)
        plt.tight_layout()
        plt.show()






