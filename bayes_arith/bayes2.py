#coding:utf-8
import re
from bayes_arith import bayes
import random
import numpy as np


# 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 文档解析
def textParse(bigString):
    listOfTokens = re.split(r"\W*", bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


#
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i,encoding="iso8859").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        try:
            wordList = textParse(open('email/ham/%d.txt' % i,encoding="iso8859").read())
        except Exception as e:
            print(i)
            raise e
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    # trainingSet=bagOfWords2VecMN(vocabList,fullText)
    trainingSet = list(range(50))
    testSet = []
    # 防止随机到重复的序号
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])
        if bayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print("The error rate is:",float(errorCount)/len(testSet))
    return float(errorCount)/len(testSet)

if __name__ == '__main__':
    rate=0.0
    for i in range(1000):
        rate+=spamTest()
    print(rate/1000)