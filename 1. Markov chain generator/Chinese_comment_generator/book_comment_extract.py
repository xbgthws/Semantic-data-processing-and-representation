import random
from random import randint

oldf = open('book_comment.txt', 'r', encoding='utf-8')  # 要被抽取的文件
newf = open('randomtext.txt', 'w', encoding='utf-8')  # 抽取的10000行写入randomtext.txt
n = 0
resultList = random.sample(range(0, 200000), 100000)  # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素

lines = oldf.readlines()
for i in resultList:
    newf.write(lines[i])
oldf.close()
newf.close()


