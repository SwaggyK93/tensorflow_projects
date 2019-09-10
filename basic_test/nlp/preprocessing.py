# _*_ coding:utf-8 _*_

# @File  : preprocessing.py
# @Author: teddyjkwang
# @Date  : 2019/9/9
# @Desc  :
import sys,os


source = open("data/source.txt","w")
target = open("data/target.txt","w")

with open('data/对联.txt', encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        source.write(line[0]+"\n")
        target.write(line[1]+"\n")
