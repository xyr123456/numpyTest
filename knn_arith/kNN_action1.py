#coding:utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import knn_arith.KNN as knn

def file2matrix(filename):
    fr=open(filename,"r")
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        listPromLine=line.strip().split("\t")
        returnMat[index,:]=listPromLine[:3]
        classLabelVector.append(listPromLine[-1])
        index+=1
    return returnMat,classLabelVector

def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,rangs,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    print(numTestVecs)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=knn.classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
            print("The right label is {0},but the forecast label is {1}".format(str(datingLabels[i]),str(classifierResult)))
    print(errorCount/float(numTestVecs))


def showFigureGraph(xList,yList,labels):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xList,yList,15*np.array([int(i) for i in labels]),15*np.array([int(i) for i in labels]))
    plt.show()

def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    # normDataSet=np.zeros(np.shape(dataSet))
    normDataSet=dataSet-np.tile(minVals,(dataSet.shape[0],1))
    normDataSet=normDataSet/np.tile(ranges,(dataSet.shape[0],1))
    return normDataSet,ranges,minVals


if __name__ == '__main__':
    # datingDataMat,datingLabels=file2matrix("datingTestSet2.txt")
    # showFigureGraph(datingDataMat[:,0],datingDataMat[:,1],datingLabels)
    datingClassTest()
