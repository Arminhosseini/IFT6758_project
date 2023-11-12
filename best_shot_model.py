import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from PIL import Image
import matplotlib.pyplot as plt
import math

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


def List_High_Corr_Features(df, threshold=0.9):

    # Find numerical features
    numerical_features = df.select_dtypes(include=['int', 'float']).columns
    df = df[numerical_features]

    # Build corr matrix
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop_features = [column for column in upper.columns if any(upper[column] > threshold)]
    return drop_features



def main():
    """
    In this script, I will:
    1. Feature selection
        a. Correlation: if the correlation between two features are larger than specific `threshold`, we will remove it.
        b. Mutual Information: identify the relationship between features and the target variable. 
            From practical visualization, we choose features, whose mutual information scores are above 0.01.
        c. Convert categorical feature into one-hot vector
        d. Standardize numerical features
    2. Decision tree
        a. Run grid search.
        b. Evaluate best model.
    """


    # ================= Hyper-parameter ==================
    path_train_csv = r"Dataset/train.csv"
    path_val_csv = r"Dataset/val.csv"

    list_features = ['period', 'x-coordinate', 'y-coordinate', 'shot_distance', 'angle', 'last_event_type', 'coor_x_last_event',\
                 'coor_y_last_event', 'time_last_event', 'distance_last_event', 'is_rebound', 'Change in shot angle', 'Speed']

    labels = 'isgoal'

    threshold_corr = 0.9
    threshold_mutual_info = 0.01

    param_grid_tree = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 5, 10]
    }
    # ====================================================


    # ================= 0. Read dataset =================
    df_train = pd.read_csv(path_train_csv)
    df_val = pd.read_csv(path_val_csv)

    df_train.dropna(inplace=True)
    df_val.dropna(inplace=True)

    X_train = df_train[list_features]
    y_train = df_train[labels]

    X_val = df_val[list_features]
    y_val = df_val[labels]


    # ================= 1. Feature selection =================
    # --- a. Remove high correlation features
    drop_features = List_High_Corr_Features(X_train, threshold=threshold_corr)
    print(f"[INFO] Drop high correlation: {drop_features}")

    X_train = X_train.drop(drop_features, axis=1)
    X_val = X_val.drop(drop_features, axis=1)


    # --- b. Extract numerical and categorical
    numerical_features = X_train.select_dtypes(include=['int', 'float']).columns
    categorical_features = [i for i in list(X_train.columns) if i not in numerical_features]

    X_train_numerical = X_train[numerical_features]
    X_train_categorical = X_train[categorical_features]

    X_val_numerical = X_val[numerical_features]
    X_val_categorical = X_val[categorical_features]


    # --- b. Mutual infomation
    mi_selector = SelectKBest(mutual_info_classif, k='all') # Calculate mutual information
    mi_selector.fit(X_train_numerical, y_train)

    selected_features = X_train_numerical.columns[mi_selector.scores_ > threshold_mutual_info]
    print(f"[INFO] Selected features (mutual information): {selected_features}")

    X_train_numerical = X_train_numerical[selected_features]
    X_val_numerical = X_val_numerical[selected_features]


    # --- c. Convert categorical feature into one-hot vector
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_train_categorical)

    X_train_categorical = encoder.transform(X_train_categorical).toarray()
    X_val_categorical = encoder.transform(X_val_categorical).toarray()


    # --- d. Standardize numerical features
    scaler = StandardScaler()
    scaler.fit(X_train_numerical)

    X_train_numerical = scaler.transform(X_train_numerical)
    X_val_numerical = scaler.transform(X_val_numerical)


    X_train = np.concatenate([X_train_numerical, X_train_categorical], axis=1)
    X_val = np.concatenate([X_val_numerical, X_val_categorical], axis=1)


    # 3. ================= Building decision tree =================

    # --- a. Grid search
    dt_classifier = DecisionTreeClassifier()

    grid_search = GridSearchCV(dt_classifier, param_grid_tree, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_dt_classifier = grid_search.best_estimator_

    # --- b. Evaluate on val test
    y_val_pred = best_dt_classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    print("Best Parameters:", best_params)
    print(f"Accuracy validation set: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)    


main()