import numpy as np
import operator



def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()      #按从小到大的顺序排好后返回新array元素所属的原array的序号
    classCount={}
    for i in range(k):      #统计距离排前k的测试样本对应标签的频次
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]




if __name__ == '__main__':
    group,labels=createDataSet()
    result=classify0([0,0],group,labels,2)
    print(result)
    returnMat=np.zeros((4,3,4))
    print(returnMat)

