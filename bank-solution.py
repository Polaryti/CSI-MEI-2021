import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

if __name__ == '__main__':
    df = pd.read_csv('bank/bank-full.csv', sep=';')

    # Data Preprocessing
    for column in df.columns:
        if df[column].dtype == type(object):
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

    X = df.drop('y', axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    # Model Training
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print('KNN train score: ', knn.score(X_train, y_train))
    print('KNN Accuracy: ', knn.score(X_test, y_test))

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    print('RF train score: ', rf.score(X_train, y_train))
    print('RF Accuracy: ', rf.score(X_test, y_test))

    svm = SVC()
    svm.fit(X_train, y_train)
    print('SVM train score: ', svm.score(X_train, y_train))
    print('SVM Accuracy: ', svm.score(X_test, y_test))
