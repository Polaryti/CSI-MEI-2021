from random import random

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (LabelEncoder, MaxAbsScaler, MinMaxScaler,
                                   Normalizer, RobustScaler, StandardScaler)
from sklearn.svm import SVC


def eda(df):
    print(df.info())
    print(df.describe())
    count_unique_values_if_categorical(df)
    df.hist(bins=50)
    plt.show()
    print(df.groupby('y').describe())
    print(df.groupby('y').agg(['mean']).unstack().plot(kind='bar'))
    plt.show()


def count_unique_values_if_categorical(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            print(col, ': ', df[col].nunique())


def base_evaluation(models, scalers, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y)
    for model in models:
        for scaler in scalers:
            print("Model: ", model, "\tScaler: ", scaler)
            if scaler is not None:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            print('Acuracy: ', model.score(X_test_scaled, y_test))
            print('Recall: ', recall_score(y_test, y_pred))
            # ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test)
            # plt.show()


def cross_evaluation(models, scalers, X, y):
    for model in models:
        for scaler in scalers:
            print("Model: ", model, "\tScaler: ", scaler)
            if scaler is not None:
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            scores = cross_validate(
                model, X_scaled, y, cv=3, scoring=('accuracy', 'recall'), n_jobs=-1)
            print('Accuracy: ', scores['test_accuracy'].mean())
            print('Recall: ', scores['test_recall'].mean())


if __name__ == '__main__':
    df = pd.read_csv('bank/bank-full.csv', sep=';')

    # eda(df)

    # Common Data Preprocessing
    for column in df.columns:
        if df[column].dtype == type(object):
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

    # iso = IsolationForest(contamination=0.05)
    # X = df.drop('y', axis=1)
    # outlier = iso.fit_predict(X)
    # df['outlier'] = outlier
    # df.drop(df[df['outlier'] == -1].index, inplace=True)
    # df.drop('outlier', axis=1, inplace=True)

    X = df.drop('y', axis=1)
    y = df['y']

    rus = RandomUnderSampler(sampling_strategy='majority')
    X, y = rus.fit_resample(X, y)

    # Models
    lr = LogisticRegression(solver='liblinear')
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=200)
    svm = SVC()

    models = [lr, knn, rf, svm]

    # Preprocessing
    scaler = StandardScaler()
    scaler2 = MinMaxScaler()
    scaler3 = MaxAbsScaler()
    scaler4 = RobustScaler()
    scaler5 = Normalizer()

    scalers = [None, scaler, scaler2, scaler3, scaler4, scaler5]

    # base_evaluation(models, X, y)
    cross_evaluation(models, scalers, X, y)
