import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston = load_boston()
print(type(boston))
print(boston.keys())
df = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
df['target'] = boston['target']
# print(df.head())
print(boston['DESCR'])
# print(df.info())
print(df.describe())

def cal_count(dataFrame):
    new_df = pd.DataFrame(index=['count'])
    for column in dataFrame:
        new_df[column] = len(dataFrame[column])
    return new_df


def cal_mean(dataFrame):
    new_df = pd.DataFrame(index=['mean'])
    for index, column in enumerate(dataFrame):
        total = 0
        for value in dataFrame[column]:
            total += value
        new_df[column] = total / len(dataFrame[column])
    return new_df


def cal_mean2(series):
    total = 0
    for i in series:
        total += i
    return total / len(series)


def cal_sqrt(variance):
    return np.sqrt(variance)


def cal_std(dataFrame):
    new_df = pd.DataFrame(index=['std'])
    for column in dataFrame:
        x_bar = cal_mean2(df[column])
        total = 0
        for i in dataFrame[column]:
            total += (i-x_bar) ** 2
        variance = total / len(dataFrame[column])
        new_df[column] = cal_sqrt(variance)
    return new_df


def cal_min(dataFrame):
    new_df = pd.DataFrame(index=['min'])
    for column in dataFrame:
        min = dataFrame[column].iloc[0]
        for value in dataFrame[column]:
            min = value if min > value else min
        new_df[column] = min
    return new_df


def cal_max(dataFrame):
    new_df = pd.DataFrame(index=['max'])
    for column in dataFrame:
        min = dataFrame[column].iloc[0]
        for value in dataFrame[column]:
            min = value if min < value else min
        new_df[column] = min
    return new_df


def cal_median(dataFrame):
    new_df = pd.DataFrame(index=['50%'])
    for column in dataFrame:
        index = (len(dataFrame[column]) + 1) / 2
        floor = (index - math.floor(index))
        if(floor == 0):
            new_df[column] = dataFrame[column].sort_values().iloc[index]
        else:
            new_df[column] = (dataFrame[column].sort_values().iloc[math.floor(index)] +
                              dataFrame[column].sort_values().iloc[math.ceil(index)]) / 2
    return new_df


def cal_25_percentiles(dataFrame):
    new_df = pd.DataFrame(index=['25%'])
    for column in dataFrame:
        index = (len(dataFrame[column]) + 1) * (25 / 100)
        floor = (index - math.floor(index))
        if(floor == 0):
            new_df[column] = dataFrame[column].sort_values().iloc[index]
        else:
            new_df[column] = (dataFrame[column].sort_values().iloc[math.floor(index)] +
                              dataFrame[column].sort_values().iloc[math.ceil(index)]) / 2
    return new_df


def cal_75_percentiles(dataFrame):
    new_df = pd.DataFrame(index=['75%'])
    for column in dataFrame:
        index = (len(dataFrame[column]) + 1) * (75 / 100)
        floor = (index - math.floor(index))
        if(floor == 0):
            new_df[column] = dataFrame[column].sort_values().iloc[index]
        else:
            new_df[column] = (dataFrame[column].sort_values().iloc[math.floor(index)] +
                              dataFrame[column].sort_values().iloc[math.ceil(index)]) / 2
    return new_df


df1 = pd.concat([cal_count(df), cal_mean(df), cal_std(
    df), cal_min(df), cal_25_percentiles(df), cal_median(df), cal_75_percentiles(df), cal_max(df)], join='outer')
print(df1)
# plt.title('Boston house prices')
# plt.scatter(data=df.sort_values(by='CRIM'),x='CRIM',y='target')
# plt.xlabel('CRIM')
# plt.scatter(data=df.sort_values(by='ZN'),x='ZN',y='target')
# plt.xlabel('ZN')
# plt.scatter(data=df.sort_values(by='INDUS'),x='INDUS',y='target')
# plt.xlabel('INDUS')
# plt.scatter(data=df.sort_values(by='CHAS'),x='CHAS',y='target')
# plt.xlabel('CHAS')
# plt.scatter(data=df.sort_values(by='NOX'),x='NOX',y='target')
# plt.xlabel('NOX')
# plt.scatter(data=df.sort_values(by='RM'),x='RM',y='target')
# plt.xlabel('RM')
# plt.scatter(data=df.sort_values(by='AGE'),x='AGE',y='target')
# plt.xlabel('AGE')
# plt.scatter(data=df.sort_values(by='DIS'),x='DIS',y='target')
# plt.xlabel('DIS')
# plt.scatter(data=df.sort_values(by='RAD'),x='RAD',y='target')
# plt.xlabel('RAD')
# plt.scatter(data=df.sort_values(by='TAX'),x='TAX',y='target')
# plt.xlabel('TAX')
# plt.scatter(data=df.sort_values(by='PTRATIO'),x='PTRATIO',y='target')
# plt.xlabel('PTRATIO')
# plt.scatter(data=df.sort_values(by='LSTAT'),x='LSTAT',y='target')
# plt.xlabel('LSTAT')
# plt.ylabel('MEDV')
# plt.show()

# sns.set_style("whitegrid")
# sns.pairplot(data=df[['CRIM','ZN','INDUS']])
# plt.show()

# sns.boxplot(data=df,x='CRIM',y='target').
# sns.boxplot(data=df,x='CHAS',y='target')
# plt.show()

# sns.violinplot(data=df,x='CHAS',y='target')
# plt.show()

# sns.distplot(df['CRIM'])
# sns.distplot(df['ZN'])
# plt.show()

# sns.jointplot(data=df,x='CHAS',y='target')
# plt.show()

# plt.bar(data=df,x='CRIM',height=5)
# plt.bar(data=df,x='CHAS',height=5)
# plt.show()

# plt.plot('CRIM',data=df,)
# plt.plot('CHAS',data=df)
# plt.legend(['CRIM','CHAS'])
# plt.show()

# sns.heatmap(df[['CHAS','target']])
# plt.show()

X = df.drop(columns="target")
y= df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 

ln = LinearRegression()
ln.fit(X_train,y_train)
print(f"Coreficiant {ln.coef_}")
print(f"Intercept {ln.intercept_}")
predictions = ln.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# df_bar = pd.DataFrame({'Y_test':y_test,'Y_train':y_train})
# df_bar.plot(kind='bar')
# plt.plot(predictions)
# plt.show()
# plt.scatter(y_test,predictions)
# sns.distplot((y_test-predictions),bins=50)
# plt.show()
# sns.heatmap(df.corr(),annot=True)
# plt.show()
result = pd.DataFrame(data= ['MAE','MSE','RMSE'],columns=['index'])
result.set_index('index',inplace=True,drop=True)
result['Data'] = [metrics.mean_absolute_error(y_test, predictions),metrics.mean_squared_error(y_test, predictions),np.sqrt(metrics.mean_squared_error(y_test, predictions))]
# Normalization 
# norm_x = (x - min_x) / (max_x - min_x)
def normalization(data):
    max_x = data.max()
    min_x = data.min()
    arr_X = []
    for x in data:
        arr_X.append(round((x - min_x) / (max_x - min_x),4))
    return arr_X
# Standardization
# z_score = (x - mean_x) / sd_x
def standardization(data):
    mean = data.mean()
    std = data.std()
    # print(f' {mean} {std}')
    arr_x = []
    for x in data:
        arr_x.append(round((x-mean) / std,4))
    return arr_x

test = pd.DataFrame(data=[94,92,78,90,90,62,87,77,74,90,98,66,61,75,76,87,95,69,79,86],columns=['Data'])
test['normalization'] = normalization(test['Data'])
test['standardization'] = standardization(test['Data'])
print(test)

def rescala_normalization(dataFrame):
    for i in dataFrame:
        dataFrame[i] = normalization(dataFrame[i])
    return dataFrame

df_norm = rescala_normalization(df.drop(columns="target"))
df_norm['target'] = df['target']
# print(df_norm.describe())
X_norm = df_norm.drop(columns="target")
y_norm = df_norm['target']
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.3, random_state=101) 

ln_norm = LinearRegression()
ln_norm.fit(X_train,y_train)
print(f"Coreficiant  normalization {ln_norm.coef_}")
print(f"Intercept normalization  {ln_norm.intercept_}")
predictions_norm = ln_norm.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions_norm))
print('MSE:', metrics.mean_squared_error(y_test, predictions_norm))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions_norm)))
result['Data_Normalization'] = [metrics.mean_absolute_error(y_test, predictions_norm),metrics.mean_squared_error(y_test, predictions_norm),np.sqrt(metrics.mean_squared_error(y_test, predictions_norm))]
print(result)
print(ln_norm.score(X_train,y_train))