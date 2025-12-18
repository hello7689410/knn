from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np



def euc_distance(i1,i2):
    return np.sqrt(sum((i1-i2)**2))

def classifier(x,y,testInstance,k):
    distances=[euc_distance(i,testInstance) for i in x]
    neighbors=np.argsort(distances)[:k]
    count=Counter(y[neighbors])
    return count.most_common()[0][0]

#提取每一个数据的两个特征，方便图像表示（在二维图像中，x轴第1个特征，y轴为第2个特征）
iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

#将数据分为train和test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=32)


#创建KNeighbors对象
model=KNeighborsClassifier(n_neighbors=3)

#开始训练,生成分类器classifier(clf)
clf=model.fit(x_train,y_train)

#检测正确性，使用test数据
count1=np.count_nonzero((clf.predict(x_test)==y_test))

#准确率
print("KNeighbors准确率：",count1/(len(x_test)))
corrects=[classifier(x_train,y_train,i,3) for i in x_test]
count2=np.count_nonzero(corrects==y_test)
print("my_knn准确率：",count2/(len(x_test)))

#绘制决策边界

h=.02  #二维平面的网格步长

x_min,x_max=x_test[:,0:1].min(),x_test[:,0:1].max()   #x轴的最大值与最小值，将特征1设为x轴，即x_test的每一个数据的第一个
y_min,y_max=x_test[:,1:].min(),x_test[:,1:].max()   #将第二个特征设为y轴
# print(x_min,x_max,y_min,y_max)

xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

print(xx,yy)
