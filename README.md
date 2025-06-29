# Heart Disease Prediction Project
This repository contains a comprehensive machine learning project focused on predicting heart disease using various classification algorithms. 
The project follows a complete data science workflow including exploratory data analysis (EDA), data preprocessing, model training, and evaluation.


## Project Workflow
1. Data Preprocessing and EDA

    - Data Cleaning: Handled missing values, addressed outliers (particularly focused on cholesterol levels where 0 values were identified as implausible), and corrected inconsistent data entries
    - Exploratory Data Analysis: Used visualizations including histograms, boxplots, and correlation matrices to understand feature distributions and relationships
    - Feature Engineering: Created derived features and applied appropriate transformations to improve model performance

2. Model Implementation
   
Multiple classification algorithms were implemented and compared:


- Logistic Regression

    - Used as a baseline model due to its interpretability
    - Explored hyperparameters through grid search
    - Analyzed coefficient values to understand feature importance

- Random Forest Classifier

    - Utilized ensemble learning approach with multiple decision trees
    - Optimized number of estimators and maximum features

- CatBoost Classifier

    - Specialized algorithm for handling categorical features
    - Optimized for performance while controlling for overfitting

- PyCaret Implementation with CatBoost

    - Leveraged PyCaret's automated machine learning capabilities specifically for CatBoost model
    - Utilized PyCaret's simplified workflow for preprocessing, model training, and evaluation
    - Streamlined hyperparameter tuning for the CatBoost model
    - Benefited from PyCaret's pipeline approach for consistent preprocessing and model deployment

## Model Evaluation
Models were evaluated using multiple metrics:

- Accuracy: Overall correctness of predictions
- Precision: Positive predictive value
- Recall/Sensitivity: True positive rate (critical for disease detection)
- F1-Score: Harmonic mean of precision and recall
- AUC-ROC: Area under the Receiver Operating Characteristic curve

Cross-validation was employed to ensure robust performance assessment and avoid overfitting.


Requirements
To run this project, the following libraries are required:

[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)

[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

[![XGBoost](https://img.shields.io/badge/XGBoost-0078D4?style=for-the-badge&logo=python&logoColor=white)](https://xgboost.readthedocs.io/)

[![CatBoost](https://img.shields.io/badge/CatBoost-FFB13B?style=for-the-badge&logo=python&logoColor=black)](https://catboost.ai/)

[![PyCaret](https://img.shields.io/badge/PyCaret-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://pycaret.org/)


