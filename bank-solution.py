import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.utils import multiclass


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
            scores = cross_val_score(
                model, X_scaled, y, cv=5, scoring='recall')
            print('Recall: ', scores.mean())


if __name__ == '__main__':
    df = pd.read_csv('bank/bank-full.csv', sep=';')

    # Common Data Preprocessing
    for column in df.columns:
        if df[column].dtype == type(object):
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

    X = df.drop('y', axis=1)
    y = df['y']

    # Models
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    k_means = KMeans()
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=200)
    svm = SVC()

    models = [lr, k_means, knn, rf, svm]

    # Preprocessing
    scaler = StandardScaler()
    scaler2 = MinMaxScaler()
    scaler3 = MaxAbsScaler()
    scaler4 = RobustScaler()
    scaler5 = Normalizer()

    scalers = [None, scaler, scaler2, scaler3, scaler4, scaler5]

    # base_evaluation(models, X, y)
    cross_evaluation(models, scalers, X, y)
