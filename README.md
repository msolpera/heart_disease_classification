# Heart Disease Prediction Project
This repository contains a machine learning project focused on predicting heart disease using various classification algorithms. 
The project follows a complete data science workflow including exploratory data analysis (EDA), data preprocessing, model training, and evaluation.


## Project Workflow
1. Data Preprocessing and EDA

    - Data Cleaning: Handled missing values, addressed outliers (particularly focused on cholesterol levels where 0 values were identified as implausible), and corrected inconsistent data entries
    - Exploratory Data Analysis: Used visualizations including histograms, boxplots, and correlation matrices to understand feature distributions and relationships
    - Feature Engineering: Created derived features and applied appropriate transformations to improve model performance

2. Model Implementation
   
Multiple classification algorithms were implemented and compared:

- Logistic Regression

- Random Forest Classifier

- CatBoost Classifier

- XGBoost 


## Model Evaluation
Models were evaluated using multiple metrics:

- Accuracy: Overall correctness of predictions
- Precision: Positive predictive value
- Recall/Sensitivity: True positive rate (critical for disease detection)
- F1-Score: Harmonic mean of precision and recall
- AUC-ROC: Area under the Receiver Operating Characteristic curve

Cross-validation was employed to ensure robust performance assessment and avoid overfitting.


### Best Model: CatBoost Classifier

After testing multiple classifiers and performing hyperparameter tuning, the **CatBoostClassifier** achieved the best performance on the test set.

**Best hyperparameters** (via RandomizedSearchCV):
- `iterations`: 200
- `learning_rate`: 0.01
- `depth`: 5
- `l2_leaf_reg`: 1
- `border_count`: 255
- `bagging_temperature`: 1.0
- `random_strength`: 0
- `subsample`: 1.0

**Test set evaluation:**

| Metric         | Value   |
|----------------|---------|
| Accuracy       | 0.8641  |
| Precision      | 0.9020  |
| Recall         | 0.8598  |
| F1-Score       | 0.8804  |
| ROC AUC        | 0.9322  |
| Specificity    | 0.8701  |
| NPV            | 0.8171  |
---

The project is structure in the following way:

main.ipynb: Jupyter notebook where load the data, train the model, and visualize the results.

src/: Contains all the modular code used in the project:

- eda.py: Performs exploratory data analysis.
- preprocessing.py: Data cleaning and Pipeline that imputes missing values, encodes categorical variables using one-hot encoding, and scales numerical variables using MinMaxScaler. Optionally, the pipeline can be fitted immediately on the provided data.
- models.py: Model training, evaluation, and performance metrics
- predict.py: Allows the user to make predictions on new data.
data/: Contains the original CSVs.

README.md: This file

requirements.txt: Libraries needed to run this code.


To install dependencies:

```bash
pip install -r requirements.txt
```



[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)

[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

[![XGBoost](https://img.shields.io/badge/XGBoost-0078D4?style=for-the-badge&logo=python&logoColor=white)](https://xgboost.readthedocs.io/)

[![CatBoost](https://img.shields.io/badge/CatBoost-FFB13B?style=for-the-badge&logo=python&logoColor=black)](https://catboost.ai/)

[![PyCaret](https://img.shields.io/badge/PyCaret-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://pycaret.org/)


