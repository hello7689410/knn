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
    data_max=cmd_list[0:50]
    data_min=cmd_list[-50:]
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
labels=get_label(r"D:\python代码\机器学习\knn\KNN_data\label.txt",3)  #得到标签
labels=[0]*50+labels    #前面50组的数组对应的标签都是0，所以加上50个0
data,data_max,data_min=load_user_cmd(r"D:\python代码\机器学习\knn\KNN_data\User3")          #得到数据


#利用data,data_max,data_min来处理每一组每一组，得到特征：(命令种类，这一组频繁与全局的频繁的重合度，这一组的不频繁命令与全局不频繁命令的重合)
features=[]     #每一组数据的特征，(f1,f2,f3)    f1:这组数据的命令种类 f2:这一组高频命令与这个用户全局的高频命令重合度  f3:这一组低频命令与全局低频命令的重合程度
for d in data:
    f=get_cmd_feature(d,data_max,data_min)
    features.append(f)

#利用train_test_split来划分
x_train,x_test,y_train,y_test=train_test_split(features,labels,train_size=0.75)

#构造KNN模型
model=KNeighborsClassifier()

#开始训练
model.fit(x_train,y_train)

#准确度
corrects=np.count_nonzero([model.predict(x_test)==y_test])
print(corrects)
print(corrects/len(x_test))
print(x_test)