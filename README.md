# KNN

一个“**谁和我长的最像就属于谁**“的算法

## 1.原理：

当我们要判断一个新样本是属于哪一个类别时	->	找到离他最近的k个点	->	看这k个点的什么类别最多	->	就属于哪一个类

例：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251201131556944.png" alt="image-20251201131556944" style="zoom:33%;" />

我们要判断红色圆圈（张三）属于哪一类（有offer或者没有offer）,就找离他最近的k个点，看这k个点有offer的多，还是myoffer的多。多的就张三就属于那一类。

##### 这里设k=3

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251201131748423.png" alt="image-20251201131748423" style="zoom:33%;" />

因为有offer的数量是2，没有offer的数量为1，所以张三属于有offer那一类。

## 2.算法流程：

1.输入一个新的样本（x_new）

2.**利用欧氏公式算出与所有的点的距离**

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251201132120799.png" alt="image-20251201132120799" style="zoom:33%;" />

3.按照算出来的距离从小到大排序

4.选出最近的k个邻居

**5.若是分类问题，选类别数量最多的那一类；若是预测问题，求这k个邻居的平均值，就是我们的预测值**

6.输出预测结果

## 3.如何确定k（使用交叉验证）

**为什么要选择k？**

答：k过大，会造成过拟合（overfitting）,考虑到更多无关的数据（与自己不像的样本）。k过小，会受到噪声（错误数据，与真正规律无关）影响。所以需要使用交叉验证来确定k。

将train_data分成多份（不能拿test_data来进行交叉验证），小部分用于validation，大部分来训练得到准确率。

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251201134003066.png" alt="image-20251201134003066" style="zoom:33%;" />

每一部分都需要被当做一次验证数据。所有部分被当做验证数据后，计算他们得到的准确度的平均值。

然后修改k，进行进行上一部分的流程。**哪个平均准确度越高，那么就确定这个k**

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251201134254076.png" alt="image-20251201134254076" style="zoom:33%;" />

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251201134236165.png" alt="image-20251201134236165" style="zoom:33%;" />

```
from sklearn import datasets  #需要用到的数据集
from sklearn.model_selection import train_test_split #将数据集分为test_data和train_data
from sklearn.neighbors import kNeighborsClassifier
```

# 4.KNN算法的使用

```
from sklearn import datasets  #需要用到的数据集
from sklearn.model_selection import train_test_split #将数据集分为test_data和train_data
from sklearn.neighbors import KNeighborsClassifier #导入KNN模型
import numpy as np  #用于处理矩阵

iris=datasets.load_iris()       #读取数据
x=iris.data #150份数据，每一份数据4给向量，代表每一份数据4个特征。shape=(150,4)
y=iris.target  #前面150份数据的label，每一份数据代表对应数据的分类，分为1,2,3类。shape=(150,)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2003)  #将x数据分为两份为x_train和x_test;将y数据分为y_train和y_test
print(y_train,y_test)

#构建KNN模型
clf=KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=None,n_neighbors=3,p=2,weights='uniform')
#algorithm（算法）:用于最近邻查询的算法   leaf_size:叶子节点大小    metric:距离度量方法   n_neighbors:k值
clf.fit(x_train,y_train) #将数据放进去，不会训练

correct=np.count_nonzero((clf.predict(x_test)==y_test)==True)   #使用model来预测每一个x_test，看与y_test相等的数量。
print(correct/len(x_test))      #计算准确率
```

## 5.创建KNN算法（KNN原理代码表示）

```
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
corrects=np.count_nonzero((predicts==y_test)==True)

#预测正确率
print("准确率")
print(corrects/len(x_test))


```

# 6.KNN项目（检测用户异常命令）

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from nltk import FreqDist   #用于统计每一条命令出现的频率
from sklearn.model_selection import train_test_split



def get_label(filename,index):      #获取标签：每一行的第一个数据就代表一组data对应的标签(从5000开始，前面5000的数据的每一组（一组100个）都为正常数据)
    labels=[]
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            line=list(map(int,line.split(' ')))
            labels.append(line[index])
    return labels

def load_user_cmd(filename):
    data=[]
    cmd_list=[]
    with open(filename,encoding='utf-8') as f:
        x=[]   #每一100份数据为一组，x为100份数据的容器
        i=0
        for line in f:
            line=line.strip('\n')   #去除'\n'
            x.append(line)
            cmd_list.append(line)
            i+=1
            if i%100==0:
                data.append(x)
                x=[]
    cmd_list=FreqDist(cmd_list)
    cmd_list=list(dict(sorted(cmd_list.items(),key=lambda x:-x[1])))
    data_max=cmd_list[0:10]
    data_min=cmd_list[-10:]
    return data,data_max,data_min

def get_cmd_feature(data,data_max,data_min):
    f1=len(set(data))
    cmd=FreqDist(data)
    cmd=dict(list(sorted(cmd.items(),key=lambda x:-x[1])))
    cmd=list(cmd.keys())
    f2=cmd[0:10]
    f3=cmd[-10:]
    f2=len(set(f2)&set(data_max))
    f3=len(set(f3)&set(data_min))
    return (f1,f2,f3)

#读取文件得到数据（所有数据，所有数据的最频繁的10个命令，所有数据最不频繁的10个命令），标签
labels=get_label(r"D:\python代码\机器学习\knn\KNN_data\label.txt",2)  #得到标签
labels=[0]*50+labels    #前面50组的数组对应的标签都是0，所以加上50个0
data,data_max,data_min=load_user_cmd(r"D:\python代码\机器学习\knn\KNN_data\User1")          #得到数据


#利用data,data_max,data_min来处理每一组每一组，得到特征：(命令种类，这一组频繁与全局的频繁的重合度，这一组的不频繁命令与全局不频繁命令的重合)
features=[]     #每一组数据的特征，(f1,f2,f3)    f1:这组数据的命令种类 f2:这一组高频命令与这个用户全局的高频命令重合度  f3:这一组低频命令与全局低频命令的重合程度
for d in data:
    f=get_cmd_feature(d,data_max,data_min)
    features.append(f)

#利用train_test_split来划分
x_train,x_test,y_train,y_test=train_test_split(features,labels,train_size=0.8,random_state=43)

#构造KNN模型
model=KNeighborsClassifier()

#开始训练
model.fit(x_train,y_train)

#准确度
corrects=np.count_nonzero([model.predict(x_test)==y_test])
print(corrects/len(x_test))
print(x_test)
```

