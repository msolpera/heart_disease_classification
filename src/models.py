from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)



def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        test_size (float): Fraction of data to use as test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def select_best_model(X_train, y_train, models, cv=5, scoring='accuracy', random_state=42):
    """
    Select the best model using cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        models (list): List of model names to evaluate
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric for evaluation
        random_state (int): Random state for reproducibility
        
    Returns:
        best_model: Trained best performing model
        results (dict): Cross-validation results for all models
    """
    
    # Define model configurations
    model_configs = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        ),
        "RNN": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            alpha=0.01,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        ),
        "Catboost": CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=random_state,
            verbose=False
        )
    }
    
    results = {}
    best_score = -1
    best_model = None
    best_model_name = None
    
    print("Evaluating models with cross-validation...")
    print("-" * 50)
    
    for model_name in models:
        if model_name not in model_configs:
            print(f"Warning: Model '{model_name}' not found in configurations. Skipping.")
            continue
            
        model = model_configs[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        
        # Store results
        results[model_name] = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"{model_name:15} | CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Track best model
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model = model
            best_model_name = model_name
    
    print("-" * 50)
    print(f"Best Model: {best_model_name} with CV Score: {best_score:.4f}")
    
    # Train the best model on full training data
    best_model.fit(X_train, y_train)
    
    return best_model, results




from sklearn.metrics import classification_report, confusion_matrix

import time

def tune_catboost_hyperparameters(X_train, y_train, 
                                  cv=5, n_iter=50, 
                                  random_state=42, verbose=True):
    """
    Perform hyperparameter tuning for CatBoost classifier.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame, optional): Validation features set
        y_val (pd.Series, optional): Validation target set
        n_iter (int): Number of iterations for RandomizedSearchCV
        random_state (int): Random state for reproducibility
        verbose (bool): Whether to print detailed results
        
    Returns:
        best_model: Tuned CatBoost model
        search_results (dict): Search results and best parameters
    """
    
    # Define hyperparameter space
    param_distributions = {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'depth': [4, 5, 6, 7, 8],
        'l2_leaf_reg': [1, 3, 5],
        'border_count': [32, 64, 128, 255],
        'bagging_temperature': [0, 0.5, 1.0, 2.0],
        'random_strength': [0, 0.5, 1.0],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Base model configuration
    base_model = CatBoostClassifier(
        random_seed=random_state,
        verbose=False,
        eval_metric='Accuracy',
        early_stopping_rounds=50 
    )
    
    print(f"Starting hyperparameter tuning for CatBoost...")
    print(f"Search space: {len(param_distributions)} parameters")
    print("-" * 60)
    
    start_time = time.time()
    
    search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_state,
            verbose=1 if verbose else 0
        )

    
    search.fit(X_train, y_train)

    
    search_time = time.time() - start_time
    
    # Get best model
    best_model = search.best_estimator_
    
    # Compile results
    search_results = {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'search_time': search_time,
        'cv_results': search.cv_results_
    }
    
    if verbose:
        print(f"\nTuning completed in {search_time:.2f} seconds")
        print("-" * 60)
        print("BEST PARAMETERS:")
        for param, value in search.best_params_.items():
            print(f"  {param:20}: {value}")
        print(f"\nBest CV Score: {search.best_score_:.4f}")

    
    return best_model, search_results


def evaluate_model(model, X, y, plot_curves=True, class_names=None):
    """
    Comprehensive evaluation of a classification model.
    
    Args:
        model: Trained classification model
        X (pd.DataFrame): features
        y (pd.Series): target
        plot_curves (bool): Whether to plot ROC and PR curves
        class_names (list): Names for classes (default: [0, 1])
        
    Returns:
        results (dict): Dictionary with all evaluation metrics
    """
    
    if class_names is None:
        class_names = ['No Heart Disease', 'Heart Disease']
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # ROC AUC (only if probabilities available)
    roc_auc = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Store results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'npv': npv,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': classification_report(y, y_pred, target_names=class_names)
    }
    
    # Print results
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"NPV:          {npv:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC:      {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              No    Yes")
    print(f"Actual   No  {tn:4d}  {fp:4d}")
    print(f"         Yes {fn:4d}  {tp:4d}")
    
    print(f"\nDetailed Classification Report:")
    print(results['classification_report'])
    
    # Plotting
    if plot_curves and y_pred_proba is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = average_precision_score(y, y_pred_proba)
        axes[2].plot(recall_curve, precision_curve, color='green', lw=2,
                    label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[2].axhline(y=y.mean(), color='red', linestyle='--', alpha=0.5,
                       label=f'Baseline ({y.mean():.3f})')
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].legend(loc="lower left")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    elif plot_curves and y_pred_proba is None:
        # Only plot confusion matrix if no probabilities
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return results