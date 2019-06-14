#!/usr/bin/env.python
# -*- coding: utf-8 -*-
import xlrd
import pandas as pd
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def load_data(file_path):
    file_data = pd.read_excel(file_path)
    pd.set_option('display.width', 180)
    pd.set_option('display.max_columns', 20)
    # print(file_data.head())
    # print(file_data.info())
    return file_data


def plot_feature(file_data):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    # Part 1
    fig = plt.figure()
    fig.set(alpha=0.8)
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
    fig.add_subplot(221)
    file_data['Exited'].value_counts().plot(kind='bar', width=0.3)
    plt.title('流失客户与未流失客户'); plt.xlabel('1为流失客户'); plt.ylabel('人数')
    fig.add_subplot(222)
    file_data['Gender'].value_counts().plot(kind='bar', width=0.3)
    plt.title('性别情况'); plt.ylabel('人数')
    fig.add_subplot(223)
    file_data['Geography'].value_counts().plot(kind='bar')
    plt.title('用户所在国家情况'); plt.ylabel('人数')
    fig.add_subplot(224)
    file_data['Age'].plot(kind='kde', color='g')
    plt.title('年龄频率图'); plt.xlabel('年龄')
    # Part 2
    gender_f = file_data['Exited'][file_data['Gender'] == 'Female'].value_counts()
    gender_m = file_data['Exited'][file_data['Gender'] == 'Male'].value_counts()
    gender_Df = pd.DataFrame({'Female': gender_f, 'Male':gender_m})
    gender_Df.plot(kind='bar', stacked=True)
    plt.title('流失客户和性别的关系'); plt.ylabel('人数')

    Geo_f = file_data['Exited'][file_data['Geography'] == 'France'].value_counts()
    Geo_s = file_data['Exited'][file_data['Geography'] == 'Spain'].value_counts()
    Geo_g = file_data['Exited'][file_data['Geography'] == 'Germany'].value_counts()
    Geo_df = pd.DataFrame({'France': Geo_f, 'Spain':Geo_s, 'Germany': Geo_g})
    Geo_df.plot(kind='bar', stacked=True)
    plt.title('流失客户和用户所在国家的关系'); plt.ylabel('人数')
    plt.show()


def feature_processing(file_data):
    std = MinMaxScaler()
    
    def change_gender(gen):
        if gen == 'Female':
            return 0
        elif gen == 'Male':
            return 1
    file_data['Gender'] = file_data['Gender'].apply(change_gender)
    file_data['CreditScore'] = std.fit_transform(np.array(file_data['CreditScore']).reshape((-1, 1)))
    file_data['Tenure'] = np.sqrt(np.array(file_data['Tenure']).reshape((-1, 1)))
    file_data['Balance'] = std.fit_transform(np.array(file_data['Balance']).reshape((-1, 1)))
    file_data['Age'] = std.fit_transform(np.array(file_data['Age']).reshape((-1, 1)))
    file_data['EstimatedSalary'] = std.fit_transform(np.array(file_data['EstimatedSalary']).reshape((-1, 1)))

    def change_Num(n):
        if n == 1:
            return 0
        else:
            return 1
    file_data['NumOfProducts'] = file_data['NumOfProducts'].apply(change_Num)
    Geography = pd.get_dummies(file_data['Geography'], prefix='Geography')
    data = pd.concat([file_data, Geography], axis=1)
    to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Geography_Spain', 'HasCrCard']
    data = data.drop(to_drop, axis=1)
    print(data.head())
    print(data.columns)
    # print(pd.value_counts(data['Exited']))
    return data


def plot_feature_score(data):
    """
    画出特征评分图
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    feature = data.columns
    select = SelectKBest(f_classif, k=5)
    select.fit(data[feature], data['Exited'])
    score = -np.log10(select.pvalues_)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(feature)), score)
    plt.title('各个特征评分'); plt.ylabel('评分')
    plt.xticks(range(len(feature)), feature, rotation=90)
    plt.show()


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


def create_model(X, y):
    """
    建立SVM模型，划分训练集，验证集合测试集，用交叉验证评估平均准确率。
    """
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    train_data, ver_data, train_label, ver_label = train_test_split(X_train_all, y_train_all,
                                                                    test_size=0.3, random_state=1)
    # log = LogisticRegression(penalty='l2', tol=1e-5, solver='liblinear', random_state=1, max_iter=100)
    # log.fit(train_data, train_label)                    #  PS：建立逻辑回归模型用于与SVM进行对比。
    log = SVC(tol=1e-5, random_state=1)
    log.fit(train_data, train_label)

    cross_result = cross_val_score(log, train_data, train_label)
    print('模型准确率', log.score(ver_data, ver_label))
    print('交叉验证准确率', cross_result.mean())
    return log


def model_fusion(model, X, y):
    """
    模型融合，使用Bagging方法进行模型融合，同样也使用训练集，验证集，测试集，
    输出融合只后的准确率，以及用在测试集上面的准确率。
    """
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    train_data, ver_data, train_label, ver_label = train_test_split(X_train_all, y_train_all,
                                                                    test_size=0.3, random_state=1)
    bag = BaggingClassifier(base_estimator=model, n_estimators=10, random_state=1, max_samples=0.2, max_features=0.9)
    bag.fit(train_data, train_label)
    print('model_fusion', bag.score(ver_data, ver_label))
    print('result-->', bag.score(X_test, y_test))
    return bag


def plot_confusion_matrix(conf_mat, labels):
    """画出混淆矩阵函数"""
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion_matrix', fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j]), horizontalalignment="center")


def plot_learning_curve(model, X, y):
    """
    使用融合后的模型画出学习曲线。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    train_sizes = np.linspace(0.1, 1.0, 20)
    train_size, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy',
                                                           n_jobs=-1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    print('test_scores_mean-->', test_scores_mean[-1])     # 输出测试评分的均值
    test_scores_std = np.std(test_scores, axis=1)
    plt.subplot(121)
    plt.plot(train_size, train_scores_mean, 'o-')
    plt.plot(train_size, test_scores_mean, 'o-')
    plt.xlabel('样本数量'); plt.ylabel('评分'); plt.ylim([0.75, 0.9])
    plt.legend(('train_scores_mean', 'test_scores_mean'))
    plt.title('学习曲线')
    plt.subplot(122)
    plt.plot(train_size, train_scores_std, 'o-')
    plt.plot(train_size, test_scores_std, 'o-')
    plt.xlabel('样本数量'); plt.ylabel('评分'); plt.ylim([0, 0.02])
    plt.legend(('train_scores_std', 'test_scores_std'))
    plt.title('误差曲线')
    plt.show()


def evaluation_model(X, y, model, labels):
    """
    模型评估函数
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_pre = model.predict(X_test)
    decision_fun = model.decision_function(X_test)
    fig_1 = plt.figure()  # 图一、画出混淆矩阵
    fig_1.set(alpha=0.8)
    conf_mat = metrics.confusion_matrix(y_test, y_pre)
    plot_confusion_matrix(conf_mat, labels)
    print('精准率--->', metrics.precision_score(y_test, y_pre))
    print('F1值--->', metrics.f1_score(y_test, y_pre))
    print('召回率--->', metrics.recall_score(y_test, y_pre))
    pre, recall, thresholds = metrics.precision_recall_curve(y_test, decision_fun)
    fig_2 = plt.figure()  # 图二、画出精准率曲线和召回率曲线
    fig_2.set(alpha=0.8)
    plt.plot(thresholds, pre[:-1])
    plt.plot(thresholds, recall[:-1])
    plt.title('precision_recall_curve', fontsize=15)
    plt.legend(['pre', 'recall'], loc='center right')
    fig_3 = plt.figure()  # 图三、画出召回率关于精准率的曲线（x轴为精准率，y轴为召回率）
    fig_3.set(alpha=0.8)
    plt.plot(pre, recall)
    plt.xlabel('precision score'); plt.ylabel('recall score')
    plt.title('recall score about precision score', fontsize=15)
    fig_4 = plt.figure()  # 图四、画出ROC曲线，并计算出AUC。
    fig_4.set(alpha=0.8)
    fpr, tpr, thr = metrics.roc_curve(y_test, decision_fun)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve', fontsize=15)
    print('AUC--->', metrics.auc(fpr, tpr))  # 计算AUC
    plt.show()


def save_model(model):
    """
    保存模型到本地
    """
    model_path = 'svm_model.sav'                      # 保存到当前文件夹
    pickle.dump(model, open(model_path, 'wb'))
    return model_path
    

def main():
    file_path = 'new_data.csv'
    file_data = load_data(file_path)
    plot_feature(file_data)
    data = feature_processing(file_data)
    plot_feature_score(data)
    X, y = split_data(data)
    X_std = standardized_data(X)
    model = create_model(X_std, y)
    # fusion_model = model_fusion(model, X_std, y)
    # plot_learning_curve(fusion_model, X_std, y)
    # evaluation_model(X_std, y, fusion_model, labels=['0', '1'])
    # save_model(model)


if __name__ == '__main__':
    main()
