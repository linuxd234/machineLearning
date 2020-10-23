#!/usr/bin/env python
# coding: utf-8
#########################################################################
# 功能：复现决策树算法
# 2020/10/23 create by linuxd
# 注：在不引入Numpy的情况下py以列表来模拟数组
#########################################################################
from math import log
import operator

########################################################################################
#1. 构造决策树

def calcShannonEnt(dataSet):
    """
    输入：数据集
    输出：香农熵（浮点数）
    功能：计算给定数据集的香农熵
    """
    numEntries=len(dataSet)
    labelCounts={}
    #将数据集中的类型及数目统计到labelCount字典中
    for featVec in dataSet:
        currentLabel=featVec[-1]
        #获取当前样本的划分类别
        if currentLabel not in labelCounts.keys():
            #类别计数
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        #p（xi）概率
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    """
    输入：数据集，特征下标，特征值
    输出：数据集（划分后子集）
    功能：给定特征及特征值进行数据集的划分
    """
    retDataSet=[]
    """
    划分好的数据集应该是这个样子的：
    [[10,5,0],
    [20,6,1],
    [5,10,1]]
    """
    for featVec in dataSet:
        #将符合条件的样本除去给定的特征放入到子数据集空间
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
       输入：数据集
       输出：特征（整形下标）
       功能：给定数据集然后挨个按特征遍历并进行试划分
       然后计算划分前后的信息增益，迭代更新最好增益及最佳特征下标
    """
    numFeatures=len(dataSet[0])-1
    #减去最后一个（最后一个是分类）获得特征总数
    baseEntropy=calcShannonEnt(dataSet)
    #分割前的熵
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):
        #按每个特征对数据集遍历
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        #获取当前特征i的分类（如性别：男/女）
        newEntropy=0.0
        for value in uniqueVals:
            #按当前特征i的每个分类计算信息增益
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
            #有几个分类就计算几次熵然后按比例加起来形成总的熵
        infoGain =baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    """
       输入：类别列表（标签）
       输出：最接近的分类
       功能：如果当特征用完不能进行划分时，该子数据集
       仍然存在2个及以上的分类，则根据出现次数返回最多的那个
    """
    classCount={}
    for vote in classList:
        #统计分类及该每个分类下的样本数
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataset,labels):
    """
       输入：数据集、标签
       输出：决策树（字典形式存储）或则是类别
       功能：对输入的数据集及标签递归构建决策树
    """
    classList=[example[-1] for example in dataset]
    #获取给定数据集的类别列表
    #classList=['a','a','a','b','b','a']
    if classList.count(classList[0])==len(classList):
        #如果该数据集里只有一类递归结束
        return classList[0]
    if len(dataset[0])==1:
        #如果数据集里存在两个以上的分类且分类标签
        # 用完了则返回最接近的作为该数据集的分类
        return  majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataset)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    #决策树（字典嵌套字典形式存储）
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in  dataset]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        #按特征的分类进行遍历，将每个分类划分一个子树
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataset,bestFeat,value),subLabels)
        #myTree[bestFeatLabel]就是一个字典
        #mytree形如{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}
    return  myTree

########################################################################################
#2. 使用决策树进行分类

def classify(inputTree, featLabels,testVec):

    #根节点（特征）的子树是按照特征的分类来构造的
    firstStr =inputTree.keys()[0]
    #获得根节点的所属特征
    #py字典的key方法返回所有键，以字符数组（list）形式
    secondDict=inputTree[firstStr]
    #获得根节点的子树（字典）
    featIndex=featLabels.index(firstStr)
    #根据根节点的分类找它在输入参数：featLabels下的下标
    for key in secondDict.keys():
        #以key（特征的每个分类）来遍历根的子树
        if testVec[featIndex]==key:
            #测试样本
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel


def main():
    dataset=[
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels=['No surfacing','flippers']

    tree=createTree(dataset,labels)
    print (tree)
if __name__ == '__main__':
    main()
    # print(__name__)