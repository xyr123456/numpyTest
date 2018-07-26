#coding:utf-8
import numpy as np
from os import listdir
from knn_arith import KNN
import re
def img2vector(filename):
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        classNameStr=int(re.match(r'^(\d)_(\d{1,4})\.txt',fileNameStr).group(1))
        hwLabels.append(classNameStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)
    testFileList=listdir('testDigits')
    errorCount=0.0
    testFileCount=0
    mTest=len(testFileList)
    for j in range(mTest):
        fileNameStr=testFileList[j]
        classNameStr=int(re.match(r'^(\d)_(\d{1,4})\.txt',fileNameStr).group(1))
        testArray=img2vector('testDigits/%s'%fileNameStr)
        testLabel=str(KNN.classify0(testArray,trainingMat,hwLabels,3))
        testFileCount+=1
        if str(classNameStr)!=testLabel:
            print("The actual num is {0},but the forecast num is {1}".format(classNameStr,testLabel))
            errorCount+=1.0
    print("The error rate is: %f"%(errorCount/float(testFileCount)))


if __name__ == '__main__':
    handwritingClassTest()