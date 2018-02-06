
# coding: utf-8

# In[1]:

import KNN


# In[1]:

import numpy as np
import operator 


# In[3]:

def creatDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


# In[4]:

group,labels=KNN.creatDataSet()


# In[82]:

def classify0(inX,dataSet,labels,k):                  #inX是你要输入的要分类的“坐标”，dataSet是上面createDataSet的array，就是已经有的，分类过的坐标，label是相应分类的标签，k是KNN，k近邻里面的k  
    dataSetSize=dataSet.shape[0]                     #dataSetSize是sataSet的行数，用上面的举例就是4行  
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet         #前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一行里有四个一样的），然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个就是坐标相减  
    sqDiffMat=diffMat**2                              #上一行得到了坐标相减，然后这里要(x1-x2)^2，要求乘方  
    sqDistances=sqDiffMat.sum(axis=1)                 #axis=1是行相加，，这样得到了(x1-x2)^2+(y1-y2)^2  
    distances=sqDistances**0.5                        #开根号，这个之后才是距离  
    sortedDistIndicies=distances.argsort()            #argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])  
    classCount={}  
    for i in range(k):  
        voteIlabel=labels[sortedDistIndicies[i]]  
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1            #get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1  
    soredClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)         #key=operator.itemgetter(1)的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序  
    return soredClassCount[0][0]             #返回类别最多的类别  


# In[17]:

classify0([0,0],group,labels,3)


# In[112]:

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #读取文件的行数。
    returnMat = np.zeros((numberOfLines,3))        #准备一个与最终矩阵行列相同的零矩阵
    classLabelVector = []                       #用于存目标变量
    fr = open(filename)                         #重新打开文件，因为上面fr.readlines()使用之后，fr会变回空列表
    index = 0                                   #行索引
    for line in fr.readlines():                 
        line = line.strip()                     #strip()默认去除空格
        listFromLine = line.split('\t')         #依据'\t'分割整行数据，让它变成列表
        returnMat[index,:] = listFromLine[0:3]  #将文件转换为矩阵的办法，[0:3]可看出目标变量不转换
        classLabelVector.append(int(listFromLine[-1]))#将目标变量放入空列表中
        index += 1
    return returnMat,classLabelVector


# In[ ]:



