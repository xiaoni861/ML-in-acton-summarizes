
# coding: utf-8

# In[27]:

import numpy as np


# In[28]:


#file2matrix函数说明:清洗数据并将数据转换为模型所需array

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         
    #读取文件的行数。
    returnMat = np.zeros((numberOfLines,3))    
    #准备一个与训练矩阵行(既不包含目标变量1、2、3）列相同的零矩阵
    classLabelVector = []                       
    #用于存目标变量
    fr = open(filename)                         
    #重新打开文件，因为上面fr.readlines()使用之后，fr会变回空列表
    index = 0                                   #行索引
    for line in fr.readlines():                 
        line = line.strip()                     #strip()默认去除空格
        listFromLine = line.split('\t')         #依据'\t'分割整行数据，让它变成列表
        returnMat[index,:] = listFromLine[0:3]  #将文件转换为矩阵的办法，[0:3]可看出目标变量不转换
        classLabelVector.append(int(listFromLine[-1]))#将目标变量放入空列表中
        index += 1
    return returnMat,classLabelVector          #输出最终array与对应的分类变量


# In[29]:

#autoNorm函数说明：归一化数据

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    #矩阵行最小值
    maxVals = dataSet.max(0)
    #矩阵行最大值
    ranges = maxVals - minVals
    #公式中的(max-min)
    normDataSet = np.zeros(np.shape(dataSet))
    #建立一个形状为目标矩阵相同的零矩阵(这步骤去掉程序也能跑）
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))#相当于公式里的(oldvalue - min)
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #数据归一化
    return normDataSet, ranges, minVals  #返回归一化数据，极值差，矩阵数值最小行


# In[20]:

#datingClassTest函数说明：算法测试

def datingClassTest(dataSet):
    hoRatio = 0.30      
    #取30%作为测试集
    datingDataMat,datingLabels = file2matrix(dataSet)       
    #读取并转换原始数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #将数据归一化
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    #测试多少次
    errorCount = 0.0
    #错误次数
    for i in range(numTestVecs):
        classifierResult = Knn.classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3) #使用K-近邻算法
        print ("预测结果: %d, 真实结果: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print ("总错误率 is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)


# In[ ]:

if __name__ == '__main__':
    datingClassTest('datingTestSet2.txt')

