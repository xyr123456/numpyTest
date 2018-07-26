#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#矩阵乘法规律：T(m,n)*T(n,o)=T(m,o)
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()    #transpose矩阵转置
    m,n=np.shape(dataMatrix)
    alpha=0.001     #照梯度移动的步长
    maxCycles=500   #迭代次数
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

#随机梯度上升算法
def stocGradAscent0(dataMatrix,classLables):
    m,n=np.shape(dataMatrix)
    alpha=0.01
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLables[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

#随机梯度上升算法改进
def stocGradAscent1(dataMatrix,classLables,numIter=500,base_alpha=0.01):
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=base_alpha+4/(1.0+j+i)
            randIndex=int(random.uniform(0,len(dataIndex)))
            index=dataIndex[randIndex]
            h=sigmoid(sum(dataMatrix[index]*weights))
            error=classLables[index]-h
            weights=weights+alpha*error*dataMatrix[index]
            del dataIndex[randIndex]
    return weights


def plotBestFit(wei):
    if type(wei).__name__!='ndarray':
        weights=wei.getA()
    else:
        weights=wei
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataArr,labelMat=loadDataSet()
    # weights=gradAscent(dataArr,labelMat)
    # plotBestFit(weights)
    weights=stocGradAscent1(np.array(dataArr),labelMat)
    plotBestFit(weights)