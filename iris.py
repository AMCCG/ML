from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# 1. ประเภทของ ML  ?
#    - ดูจาก Input เป็น Supervised Learning เพราะข้อมูลมี labeled
#    - ดูจาก Output เป็น Classification problem เพราะ output ที่ได้เป็น class
# 2. ทำความเข้าใขกับข้อมูล
#    - Analyze the Data
#       - สถิติเชิงพรรณนา
#       - สร้างกราฟ
#    - Process the Data
#       - pre-processing (Feature Extraction) คัดเลือก Data ที่จะนำมาเป็น feature ตัด data ไม่ใช้หรือไม่มีประโยชน์ออกให้หมด
#       - profiling
#       - cleansing การตรวจสอบความถูกต้องของข้อมูลทั้งหมด
#    - Transform the data
#       - การแปลงข้อมูลดิบไปเป็นข้อมูลที่เราจะนำมาใช้สอน ml
# 3. หาอัลกอลิทึมที่ใช้ได้ดีกับข้อมูล
#    -  ตัดสินใจจาก
#       - The accuracy of the model  ความเเม่นยำของ model
#       - The interpretability of the model  ความสัมพันธ์กันของ data กับ model
#       - The complexity of the model  ความซับซ้อนของ model
#       - The scalability of the model  ความสามารถในการปรับขนาดของ model
#       - How long does it take to build, train, and test the model?  เวลาที่ใช้ไปในการสร้างเเละสอน model
#       - How long does it take to make predictions using the model?  เวลาที่ใช้ไปในทำนายของ model
#       - Does the model meet the business goal?  model ตอบโจทย์กับธุรกิจหรือป่าว
# 4. นำ machine learning มาใช้งาน
#    -  machine learning pipeline
# 5. เพิ่มประสทิธิภาพให้กับ Hyperparameter
#    - มีสามวิธีคือ grid search, random search, Bayesian optimization
#
# Machine Learning Task
#   - Supervised learning
#       - Regression
#       - Classification
#   - Unsupervused learning
#       - Cluster
#   - Reinforcement learning
#
# Machine Learning Algorithms
#   - Linear Regression
#       - ข้อมูลเชิงปริมาณ quantitative
#       - หาความสัมพันธ์ของตัวแปร X ที่เป็นตัวแปรอิสระกับ Y ที่เป็นตัวแปรตาม
#       - loss function ด้วย mean squared error (MSE) , mean absolute error (MAE)
#   - Logistic Regression
#       - เป็นประเภท classification เป็น subset ของ supervised learning
#       - output เป็น categorical หรือ binary
#   - K-means
#       - เป็นประเภท clustering เป็น subset ของ unsupervised learning
#       - จัดกลุ่มข้อมูลที่เข้ามาว่าจะอยู่ในกลุ่มไหนตามความเหมือนหรือคล้ายกัน
#       - K คือจำนวน cluster
#   - K-nearest-neigbors
#       - เป็นประเภท classification เป็น subset ของ supervised learning
#       - เป็นการจัดกลุ่มข้อมูลเมื่อมี data ใหม่เข้ามา data มีค่าใกล้เคียงกับค่าใดก็จะอยู่กลุ่มนั้น
#       - K คือค่าที่ใช้ในการหนดการ Vote
#   - Support Vector Machines
#       - เป็นประเภท classification
#       - เเบ่งเป็น 2 class ด้วยเส้นตรง
#   - Random Forest
#       - เป็นประเภท regression,classification ป็น subset ของ Supervised learning
#       - Rule base คือต้องสร้างกฏ if-else จาก feature ซิ่งมา decision tree หลายๆ อัน
#       - เป็น model แบบ ensemble คือใช้ model หลายๆ model มาประกอบกันเป็น model ที่ซับซ้อน
#   - Neural networks


iris = load_iris()
print(iris.keys())
# print(iris['DESCR'])
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']


def getTargetNames(index):
    return iris['target_names'][index]


df['target_names'] = df['target'].apply(getTargetNames)
print(df.head(150))
print(iris['target_names'])
# print(df.info())
# print(iris['feature_names'])
# print(df.describe())

# print(pd.crosstab(index=df['sepal length (cm)'], columns='frequency').apply(
#     lambda n: (n / len(df['sepal length (cm)'])) * 100, axis=1))
# plt.hist(x='sepal length (cm)', bins=30, data=df,
#          label='sepal length (cm)', alpha=0.7)
# plt.hist(x='sepal width (cm)', bins=30, data=df,
#          label='sepal width (cm)', alpha=0.7)
# plt.hist(x='petal length (cm)', bins=30, data=df,
#          label='petal length (cm)', alpha=0.7)
# plt.hist(x='petal width (cm)', bins=30, data=df,
#          label='petal width (cm)', alpha=0.7)
# plt.legend(loc="upper right")
# plt.title("Histogram")
# plt.show()
# print("Mean ",df['sepal length (cm)'].mean())


def getMean(data):
    print("----- Mean -----")
    for c in data.columns:
        m = 0
        for value in data[c]:
            m += value
        print(f"{c} : {m/len(data[c])}")


def getMedian(data):
    print("----- Median -----")
    for c in data.columns:
        data[c] = data[c].sort_values()
        print(f"{c} : { (len(data[c]) +1) /2}")


def getMode(data):
    print("----- Mode -----")
    for c in data.columns:
        d = pd.crosstab(index=data[c], columns='Mode').sort_values(
            by='Mode', ascending=False)
        print(f"{c} : {d['Mode'].iloc[0]}")


def getRange(data):
    print("----- Range -----")
    for c in data.columns:
        data[c] = data[c].sort_values()
        print(f"{c} : { (df[c].max() - df[c].min() )}")


def getStandardDeviation(data):
    print("----- StandardDeviation -----")
    for c in data.columns:
        n = len(data[c])
        mean = data[c].mean()
        sum_x = ((data[c] - mean) ** 2).sum()
        x = sum_x / (n - 1)
        std = np.sqrt(x)
        print(f"{c} :  { (std) } ")


def getQuartileDeviation(data):
    print("----- QuartileDeviation -----")
    for c in data.columns:
        n = len(data[c])
        data[c] = data[c].sort_values()
        q1 = (1 * (n+1)) / 4
        q3 = (3 * (n+1)) / 4
        print(f'{q1} {q3}')
        if type(q1) == float:
            q1 = (data[c].loc[math.ceil(q1)-1] +
                  data[c].loc[math.floor(q1)]-1) / 2
        else:
            q1 = data[c].loc[q1]
        if type(q3) == float:
            q3 = (data[c].loc[math.ceil(q3)-1] +
                  data[c].loc[math.floor(q3)-1]) / 2
        else:
            q3 = data[c].loc[q3]
        print(f'{c} : {q1} {q3} =  {(q3-q1) / 2}')


# getMean(df)
# getMedian(df)
# getMode(df)
# getRange(df)
# getStandardDeviation(df)
# getQuartileDeviation(df)

# K-nearestn-neigbors

X = df.drop(columns=["target", "target_names"], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
# for i in np.arange(1, 20):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     prediction = knn.predict(X_test)
#     print(f"{i} : {knn.score(X_train, y_train)}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print(knn.score(X_train, y_train))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction,
                            target_names=iris['target_names']))

print(y_train.iloc[[1]])
print(knn.predict(X_train.iloc[[1]]))

# # LogisticRegression

# log = LogisticRegression()

# log.fit(X_train, y_train)
# print(log.coef_)
# prediction_log = log.predict(X_test)
# print(confusion_matrix(y_test, prediction_log))
# print(classification_report(y_test, prediction_log,
#                             target_names=iris['target_names']))
