from collections import Counter     #用于计算矩阵中每一个数据出现多少次
from sklearn import datasets    #导入数据集
from sklearn.model_selection import train_test_split    #分隔数据集为train_data和test_data
import numpy as np  #用于矩阵计算

#计算欧氏函数距离的函数
def euc_distance(instance1,instance2):  #计算两个样本的距离
    distance=np.sqrt(sum((instance1-instance2)**2))     #欧氏函数,计算(a1,b1,c1,d1)和(a2,b2,c2,d2)之间的距离
    return distance

#判断一个数据样本的类别，使用欧氏函数的距离进行排序，选取附近k个数据出现最多的类别。例0,0,1 那么这个类别就是0
def knn_classify(x,y,testInstance,k):    #x,y:所有的除testInstance数据和标签，testInstance:被测量点，我们要算所有点离这个点距离
    distances=[euc_distance(i,testInstance) for i in x] #所有点离testInstance的距离
    kneighbors=np.argsort(distances)[:k]    #通过距离来对以前的索引进行排序，例：[0.3,0.2,0.1],返回的就是[2,1,0]
    count=Counter(y[kneighbors])    #计算最近的k个样本，每一个类别的数量。[1,2,3,2]返回{1:1,2:2,3:1}
    return count.most_common()[0][0] #most_commom使dict按照值从大大小排序，所以返回：{2:2,1:1,3:1},所以去第一个值的键（类别）[0][0]

#导入数据集
iris=datasets.load_iris()
data=iris.data      #数据集
target=iris.target  #数据的label
print(data,target)

#分隔数据集用于train和test
x_train,x_test,y_train,y_test=train_test_split(data,target,random_state=2003)

#使用KNN_classification对每一个样本进行预测来分类
predicts=[knn_classify(x_train,y_train,i,3) for i in x_test]

#比较预测值和真实值,看x_test运用这个模型得到的类别，会与y_test的差别,0为预测错误，1为预测正确
corrects=np.count_nonzero((predicts==y_test))

#预测正确率
print("准确率")
print(corrects/len(x_test))

