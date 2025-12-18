# -*- coding:utf-8 -*-

import numpy as np
from nltk.probability import FreqDist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics

from 机器学习.knn.KNN_data.user_behavior_analysis_knn import features

# ============================
# 与参考代码完全一致的逻辑
# ============================

# 一组样本的命令数
N = 100


def get_label(filename, index=0):
    labels = []
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            labels.append(int(line.split()[index]))
    return labels


def load_user_cmd(filename):
    """
    按参考代码逻辑：
    - 每 100 行作为一个 block
    - dist 保存所有命令，用于全局频率排序
    """
    cmd_block_list = []
    dist = []
    x = []
    i = 0

    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            x.append(line)
            dist.append(line)
            i += 1

            # 满 100 行构成一个 block
            if i == 100:
                cmd_block_list.append(x)
                x = []
                i = 0

    cmd_list = FreqDist(dist)
    cmd_list = dict(sorted(cmd_list.items(), key=lambda x: -x[1]))
    fdist = list(cmd_list.keys())  # 转成 list 才能切片
    dist_max = set(fdist[0:50])  # 高频命令
    dist_min = set(fdist[-50:])  # 低频命令

    return cmd_block_list, dist_max, dist_min


def  get_user_cmd_feature(dat,data_max,data_min):
    features=[]
    for data in dat:
        f1=len(set(data))
        cmd=FreqDist(data)
        cmd=dict(list(sorted(cmd.items(),key=lambda x:-x[1])))
        cmd=list(cmd.keys())
        f2=cmd[0:10]
        f3=cmd[-10:]
        f2=len(set(f2)&set(data_max))
        f3=len(set(f3)&set(data_min))
        features.append([f1,f2,f3])
    return features


# ================================
# 主程序（结构与参考代码完全保持一致）
# ================================
if __name__ == '__main__':

    # 读取用户命令
    user_cmd_list, dist_max, dist_min = load_user_cmd(r"D:\python代码\机器学习\knn\KNN_data\User3")

    # 抽取特征
    user_cmd_feature = get_user_cmd_feature(user_cmd_list, dist_max, dist_min)

    # 读取标签（使用第 2 列）
    labels = get_label(r"D:\python代码\机器学习\knn\KNN_data\label.txt", 3)

    # 按参考代码逻辑：前 50 块默认标记为 0
    y = [0] * 50 + labels

    # 分训练集与测试集
    x_train = user_cmd_feature[0:N]
    y_train = y[0:N]

    x_test = user_cmd_feature[N:150]
    y_test = y[N:150]

    # KNN 模型
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)

    y_predict = knn.predict(x_test)

    score = np.mean(y_predict == y_test) * 100

    print("真实标签:", y_test)
    print("预测标签:", y_predict)
    print("准确率:", score)

    print("\n分类报告：")
    print(classification_report(y_test, y_predict))

    print("混淆矩阵：")
    print(metrics.confusion_matrix(y_test, y_predict))
