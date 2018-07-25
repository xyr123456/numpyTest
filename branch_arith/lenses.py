#coding:utf-8
import branch_arith.branch as br

fr=open('lenses.txt')
lenses=[inst.strip().split("\t") for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=br.createTree(lenses,lensesLabels)
print(lensesTree)
br.createPlot(lensesTree)


