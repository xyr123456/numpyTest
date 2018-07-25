#coding:utf-8
from math import log
import numpy as np
import operator
import matplotlib.pyplot as plt
# import matplotlib
import pickle

# matplotlib.use('qt4agg')
# matplotlib.rcParams['font.sans-serif']=['SimHei']
# matplotlib.rcParams['font.family']='sans-serif'
# matplotlib.rcParams['axes.unicode_minus']=False

#计算数据集香浓熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if labelCounts.get(currentLabel,0)==0:
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonBnt=0.0
    for label in labelCounts.keys():
        prob=labelCounts.get(label) / float(numEntries)
        shannonBnt-=prob*log(prob,2)
    return shannonBnt




#划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reduceFeatVec=featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择最好的数据集（最后一列为最终所需的标签列，故不需要参与最小信息熵的评判）
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)    #创建唯一的分类标签列表
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy     #信息熵减少量
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

#获取数组中出现频次最高的元素
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    #注意此步返回的是一个二元元组
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter[1],reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]     #此步不能省，因为列表是引用的关系，需要通过切片的方式复制新的list
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")


def plotNode(nodeText,centerPt,parentPt,nodeType):
    #createPlot.ax1这种方法是为了定义全局变量，这样可以同时在多个函数里使用
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

#分类
def plotMidText(cntrPt,parentPt,txtString):
   xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
   yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
   createPlot.ax1.text(xMid,yMid,txtString)

#递归绘制树结构
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD


def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

#获取树的叶子节点个数
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

#获取树的深度
def getTreeDepth(myTree):
    maxDepth=0
    secondDict=myTree[list(myTree.keys())[0]]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  #判断是否为字典类型
            depth=getTreeDepth(secondDict[key])+1
        else:
            depth=1
        if maxDepth<depth:
            maxDepth=depth
    return maxDepth

#分类函数
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    result=secondDict.get(testVec[featIndex],'no result')
    if type(result).__name__=='dict':
        classLabel=classify(result,featLabels,testVec)
    else:
        classLabel=result
    return classLabel


#存储决策树
def storeTree(inputTree,filename):
    fw=open(filename,"w")
    pickle.dump(inputTree,fw)
    fw.close()

#读取决策树
def grabTree(filename):
    fr=open(filename)
    return pickle.load(fr)


if __name__ == '__main__':
    # myDate,labels=createDataSet()
    # myTree=createTree(myDate,labels)
    # print(myTree)
    # # createPlot(myTree)
    # res=classify(myTree,['no surfacing','flippers'],[1,1])
    # print(res)
    # fr=open()
    pass