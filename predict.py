import pandas as pd
import joblib

def predict_from_csv(csv_path, model_path="model_heart_disease_pipeline.pkl", columns_path="model_columns.pkl"):
    """
    Loads a CSV file, validates its columns, and returns predictions using a trained pipeline.

    Args:
        csv_path (str): Path to the CSV file containing new data.
        model_path (str): Path to the .pkl file with the trained pipeline (preprocessing + model).
        columns_path (str): Path to the .pkl file containing the list of expected column names.

    Returns:
        np.ndarray: Model predictions for the input data.
    """

    # Load the trained pipeline and expected column names
    pipeline = joblib.load(model_path)
    expected_columns = joblib.load(columns_path)

    # Load new data
    X_new = pd.read_csv(csv_path)

    # Validate column names and order
    if list(X_new.columns) != list(expected_columns):
        raise ValueError(f"Column mismatch:\nExpected: {expected_columns}\nReceived: {list(X_new.columns)}")

    # Make predictions
    predictions = pipeline.predict(X_new)
    return predictions


predictions = predict_from_csv("new_data.csv")
pd.DataFrame({"Prediction": predictions}).to_csv("predictions.csv", index=False)