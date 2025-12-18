from sklearn import datasets  #需要用到的数据集
from sklearn.model_selection import train_test_split #将数据集分为test_data和train_data
from sklearn.neighbors import KNeighborsClassifier #导入KNN模型
import numpy as np  #用于处理矩阵
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()       #读取数据
x=iris.data #150份数据，每一份数据4给向量，代表每一份数据4个特征。shape=(150,4)
y=iris.target  #前面150份数据的label，每一份数据代表对应数据的分类，分为1,2,3类。shape=(150,)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2003)  #将x数据分为两份为x_train和x_test;将y数据分为y_train和y_test
print(y_train,y_test)

#构建KNN模型
clf=KNeighborsClassifier(n_neighbors=3)
#algorithm（算法）:用于最近邻查询的算法   leaf_size:叶子节点大小    metric:距离度量方法   n_neighbors:k值
clf.fit(x_train,y_train) #将数据放进去，不会训练

correct=np.count_nonzero((clf.predict(x_test)==y_test)==True)   #使用model来预测每一个x_test，看与y_test相等的数量。
print(correct/len(x_test))      #计算准确率