import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(
    "https://raw.githubusercontent.com/mrpeerat/Machine_Learning_0-100/master/04%20Logistic%20Regression/titanic_data.csv")
print(df.shape)
print(df.head())
print(df.isnull().sum())


def imputerAge():
    print(df[df['Age'].isnull()])
    age = df['Age'].values
    age = np.reshape(age, (-1, 1))
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp.fit(age)
    df['Age'] = imp.transform(age)
    print(df[df['Age'].isnull()])


def imputerAge2():
    print(df[df['Age'].isnull()])
    print(df['Age'].value_counts())
    df['Age'] = df['Age'].apply(lambda x: 24 if math.isnan(x) else x)
    print(df[df['Age'].isnull()])


def displotAge():
    sns.distplot(df['Age'])
    plt.show()


def pairplot():
    sns.pairplot(df)
    plt.show()


print(df.describe())
imputerAge2()
# displotAge()
# pairplot()
print(df.corr())

# Evaluate Model


def predict1():
    X = df[['Pclass', 'Fare']]
    y = df['Survived']
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(lr.score(X_train, y_train))
    print(accuracy_score(y_test, y_pred))
    ans = lr.predict_proba([[3, 0]])
    print(ans[:])
    ans = lr.predict_proba([[2, 31]])
    print(ans[:])
    ans = lr.predict_proba([[1, 75]])
    print(ans[:])
    ans = lr.predict_proba([[1, 300]])
    print(ans[:])


predict1()

# Data Preprocessing


def dummy_Sex():
    return pd.concat([df, pd.get_dummies(df['Sex'], dummy_na=False, prefix='Sex')], axis=1).drop(['Sex'], axis=1)


df = dummy_Sex()
print(df)


def predict2():
    X = df[['Pclass', 'Fare', 'Sex_female', 'Sex_male']]
    y = df['Survived']
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(lr.score(X_train, y_train))
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


predict2()

# Scale


def predict3():
    X = df[['Pclass', 'Fare', 'Sex_female', 'Sex_male', 'Age']]
    y = df['Survived']
    scale = StandardScaler()
    X = scale.fit_transform(X)
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(lr.score(X_train, y_train))
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    parameters = {'C': np.arange(1, 10, 0.5)}
    lr_best = GridSearchCV(lr, parameters, cv=5)
    lr_best.fit(X_train, y_train)
    print(lr_best.best_estimator_)
    y_pred = lr_best.predict(X_test)
    print(lr.score(X_train, y_train))
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


predict3()

print(df.corr())


def dummy_Embarked():
    return pd.concat(
        [df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)


df = dummy_Embarked()
print(df.corr())


def predict4():
    X = df[['Pclass', 'Fare', 'Sex_female', 'Sex_male', 'Age',
            'Parch', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp']]
    y = df['Survived']
    scale = StandardScaler()
    X = scale.fit_transform(X)
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(lr.score(X_train, y_train))
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    parameters = {'C': np.arange(1, 10, 0.5)}
    lr_best = GridSearchCV(lr, parameters, cv=5)
    lr_best.fit(X_train, y_train)
    print(lr_best.best_estimator_)
    y_pred = lr_best.predict(X_test)
    print(lr.score(X_train, y_train))
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


predict4()
