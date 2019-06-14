#!/usr/bin/env.python
# -*- coding: utf-8 -*-

# 计划使用自己所熟悉的几种算法，包括KNN、逻辑回归、朴素贝叶斯、决策树、支持向量机5种。


import warnings
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")


def load_data(file_path):
    file_data = pd.read_excel(file_path)
    pd.set_option('display.width', 180)
    pd.set_option('display.max_columns', 20)
    print(file_data.head())
    print(file_data.info())
    return file_data


def feature_processing(file_data):
    std = MinMaxScaler()

    def change_gender(gen):
        if gen == 'Female':
            return 0
        elif gen == 'Male':
            return 1

    def change_Num(n):
        if n == 1:
            return 0
        else:
            return 1

    file_data['Gender'] = file_data['Gender'].apply(change_gender)
    file_data['CreditScore'] = std.fit_transform(np.array(file_data['CreditScore']).reshape((-1, 1)))
    file_data['Tenure'] = np.sqrt(np.array(file_data['Tenure']).reshape((-1, 1)))
    file_data['Balance'] = std.fit_transform(np.array(file_data['Balance']).reshape((-1, 1)))
    file_data['Age'] = std.fit_transform(np.array(file_data['Age']).reshape((-1, 1)))
    file_data['EstimatedSalary'] = std.fit_transform(np.array(file_data['EstimatedSalary']).reshape((-1, 1)))

    file_data['NumOfProducts'] = file_data['NumOfProducts'].apply(change_Num)
    Geography = pd.get_dummies(file_data['Geography'], prefix='Geography')
    data = pd.concat([file_data, Geography], axis=1)
    to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Geography_Spain', 'HasCrCard']
    data = data.drop(to_drop, axis=1)
    print(data.head())
    # print(data.columns)
    # print(pd.value_counts(data['Exited']))
    # print(data)
    return data


def split_data(data):
    """
    切分数据，将标签列和其他数据分割
    """
    y = np.array(data['Exited'])
    data = data.drop('Exited', axis=1)
    X = data.values
    return X, y


def standardized_data(X):
    """
    标准化数据函数
    """
    std = StandardScaler()
    X_std = std.fit_transform(X)
    return X_std
# # 在使用算法的时候可以使用对X数据进行标准化的数据也可以使用为标准化的数据。


# def algorithm(X, y):
#     """
#     :param X:
#     :param y:
#     :return:
#     """
#     accuracy = []
#     accuracy_1 = []
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#     alg = [KNeighborsClassifier, LogisticRegression, GaussianNB, DecisionTreeClassifier, SVC]
#     for i in alg:
#         model = i()
#         model.fit(X_test, y_test)
#         acc = model.score(X_train, y_train)
#         accuracy.append(acc)
#     knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
#     knn.fit(X_train, y_train)
#     # print('knn=', knn.score(X_test, y_test))
#     accuracy_1.append(knn.score(X_test, y_test))
#     lr = LogisticRegression(penalty='l2', tol=1e-5, solver='liblinear', random_state=1, max_iter=100)
#     lr.fit(X_train, y_train)
#     # print('lr=', lr.score(X_test, y_test))
#     accuracy_1.append(lr.score(X_test, y_test))
#     gn = GaussianNB()
#     gn.fit(X_train, y_train)
#     # print('gn=', gn.score(X_test, y_test))
#     accuracy_1.append(gn.score(X_test, y_test))
#     dt = DecisionTreeClassifier(min_samples_split=300)
#     dt.fit(X_test, y_test)
#     # print('dt=', dt.score(X_train, y_train))
#     accuracy_1.append(dt.score(X_train, y_train))
#     svc = SVC(C=2.0, tol=1e-6, random_state=1)
#     svc.fit(X_test, y_test)
#     # print('svm=', svc.score(X_train, y_train))
#     accuracy_1.append(svc.score(X_train, y_train))
#     # print(accuracy)
#     # print(accuracy_1)
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#     alg_str = ['KNeighborsClassifier', 'LogisticRegression', 'GaussianNB', 'DecisionTreeClassifier', 'SVC']
#
#     plt.bar(x=alg_str, height=accuracy, width=0.5, label='1')
#     plt.title('未经过调参的算法准确率')
#     plt.show()
#     plt.bar(x=alg_str, height=accuracy_1, width=0.5, label='2')
#     plt.title('调参之后的算法准确率')
#     plt.show()
def choose_algorithm(X, y):
    """
    :param X:
    :param y:
    :return:
    """
    models = list()
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('LR', LogisticRegression()))
    models.append(('NB', GaussianNB()))
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    result = []
    names = []
    kfo = KFold(n_splits=10, shuffle=True, random_state=1)
    for name, model in models:
        cro_result = cross_val_score(model, X, y, scoring='accuracy', cv=kfo)
        result.append(cro_result)
        names.append(name)
        msg = ("{}:{:3f}\t({:3f})".format(name, cro_result.mean(), cro_result.std()))
        print(msg)
    fig = plt.figure()
    fig.suptitle('Algorithm to compare')
    ax = fig.add_subplot(111)
    plt.boxplot(result)
    plt.ylabel('accuracy')
    ax.set_xticklabels(names)
    plt.show()


def main():
    file_path = 'new_data.csv'
    file_data = load_data(file_path)
    all_data = feature_processing(file_data)
    data = split_data(all_data)
    X_std = standardized_data(data[0])
    choose_algorithm(X_std, data[1])
    # print(data[0])
    # print(data[1])


if __name__ == '__main__':
    main()
