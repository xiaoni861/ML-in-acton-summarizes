
# coding: utf-8

# In[2]:

import numpy as np
import operator 


# In[4]:



#含①②这些数字标识的为第四部分补充知识点 
#creatDataSet函数说明：创建训练数据集

def creatDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) 
    labels=['A','A','B','B'] #分类变量A与B 
    return group,labels


# In[5]:

#classify0函数说明：实现K-近邻算法

def classify0(inX,dataSet,labels,k): 
    #inX预测数据，dataSet训练数据，labels分类变量，k选择多少个邻居              
    dataSetSize=dataSet.shape[0] 
    #返回矩阵的行数，shape[1]返回列                       
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet  
    #将inX的行数与dataSet一致然后相减（前提inX与dataSet列数一致）  
    #
    sqDiffMat=diffMat**2   
    sqDistances=sqDiffMat.sum(axis=1)
    #矩阵.sum(axis=1)行相加.(axis=0列相加)                 
    distances=sqDistances**0.5                        
    sortedDistIndicies=distances.argsort()
    #将距离从小到大排序，输出对应索引
    #           
    classCount={}  
    for i in range(k):  #k-近邻的k,用于决定选择多少个邻居
        voteIlabel=labels[sortedDistIndicies[i]]  #将类别标签赋值
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #计算各类别出现的频次            
    soredClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #将类别依据频次排序，       
    return soredClassCount[0][0]   #输出分类结果


# In[11]:

if __name__ == '__main__':
    group, labels = creatDataSet()
    #创建数据集
    result=classify0([0,0],group,labels,3)
    print(result)

