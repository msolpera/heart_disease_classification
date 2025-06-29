import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def transform_features(df, cat_features, num_features):
    """
    Transforms the cleaned dataset into features and target for modeling.

    Args:
        df (pd.DataFrame): Cleaned input DataFrame with required columns.
        cat_features (list Python): categorical features columns
        num_features (list Python): numerical features columns
    Returns:
        X (pd.DataFrame): Feature matrix ready for training.
        y (pd.Series): Target variable.
    """

    df_feat = df.copy()
    encoder = OneHotEncoder(sparse_output=False)
    enc = encoder.fit_transform(df_feat[cat_features])
    df_encoded = pd.DataFrame(
                enc,
                columns=encoder.get_feature_names_out(cat_features)
                )
    df_encoded = pd.concat([df_feat.drop(cat_features, axis=1), df_encoded], axis=1)

    y = df_encoded["HeartDisease"]
    X = df_encoded.drop("HeartDisease", axis=1)  
    
    return X, y