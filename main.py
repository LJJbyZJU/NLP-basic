# ===========================================================================
# Default handling
import pandas as pd
import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# 判断是否有缺失值数据 - isnull，notnull
# isnull：缺失值为True，非缺失值为False
# notnull：缺失值为False，非缺失值为True

# 创建数据
s = pd.Series([12, 33, 45, 23, np.nan, np.nan, 66, 54, np.nan, 99])
df = pd.DataFrame({'value1': [12, 33, 45, 23, np.nan, np.nan, 66, 54, np.nan, 99, 190],
                   'value2': ['a', 'b', 'c', 'd', 'e', np.nan, np.nan, 'f', 'g', np.nan, 'g']})
print(s.isnull())  # Series直接判断是否是缺失值，返回一个Series
print(df.notnull())  # Dataframe直接判断是否是缺失值，返回一个Series
print(df['value1'].notnull())  # 通过索引判断
print('=' * 20)

# 筛选非缺失值
s2 = s[s.isnull() == False]
df2 = df[df['value2'].notnull()]  # 注意和 df2 = df[df['value2'].notnull()] ['value1'] 的区别
print(s2)
print(df2)

# --------------------------------------------------------
# 删除缺失值
s.dropna(inplace=True)
df2 = df['value1'].dropna()
print(s)
print(df2)
# drop方法：可直接用于Series，Dataframe
# 注意inplace参数，默认False → 生成新的值

# -------------------------------------------------------
# 缺失值的处理，均值、中位数、众数插补
# 创建数据
s = pd.Series([1, 2, 3, np.nan, 3, 4, 5, 5, 5, 5, np.nan, np.nan, 6, 6, 7, 12, 2, np.nan, 3, 4])
# 分别求出均值/中位数/众数
u = s.mean()  # 均值
me = s.median()  # 中位数
mod = s.mode()  # 众数
print('均值为：%.2f, 中位数为：%.2f' % (u, me))
print('众数为：', mod.tolist())
print('------')
# 用均值填补
s.fillna(u, inplace=True)
print(s)

# ====================================================================
# 离群点处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
data = pd.Series(np.random.randn(10000) * 100)
x = pd.Series([i + 1 for i in range(10000)])
# 正态性检验
u = data.mean()  # 计算均值
std = data.std()  # 计算标准差
print('均值为：%.3f，标准差为：%.3f' % (u, std))
print('------')
# 筛选出异常值error、剔除异常值之后的数据data_c
error_index = np.abs(data - u) > 3 * std
data_c_index = np.abs(data - u) <= 3 * std
error = data[np.abs(data - u) > 3 * std]
data_c = data[np.abs(data - u) <= 3 * std]
print('异常值共%i条' % len(error))

plt.title('abnormal %i pieces' % len(error))
plt.scatter(x[error_index], data[error_index], s=5, c='r')
plt.scatter(x[data_c_index], data[data_c_index], s=5, c='b')
plt.show()

# ==================================================
# 最小最大规范化
from sklearn import preprocessing  # 导入处理库
import numpy as np

X = np.array([[1, -1, 2, 3],  # 原始数据
              [2, 0, 0, 1],
              [0, 1, -1, 2]])
min_max_scaler = preprocessing.MinMaxScaler()  # 按列进行归一化处理
X_minmax = min_max_scaler.fit_transform(X)
print(X_minmax)

# ---------------------------------------------------
# Z-得分规范化
from sklearn import preprocessing
import numpy as np

X = np.array([[1, -1, 2, 3],  # 原始数据
              [2, 0, 0, 1],
              [0, 1, -1, 2]])
X_scaled = preprocessing.scale(X)  # z得分标准化后

print(X_scaled)
print(X_scaled.mean(axis=0))  # 按列求均值
print(X_scaled.std(axis=0))  # 按列求标准差


# ==================================================
# TF-IDF
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict


def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if word in idfDict:
                if val > 0:
                    idfDict[word] += 1
            else:
                if val > 0:
                    idfDict[word] = 1
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
    return idfDict


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


import os
from string import punctuation as punc
import re

with open("./example/51120", "r") as f:
    essey1 = f.read()
with open("./example/51121", "r") as f:
    essey2 = f.read()

punc += "\n"
# essey1和essey2经过处理以后，将文章中的单词使用空格隔开，且大写变为小写，方便后续处理
essey1 = re.sub(r"[{}]+".format(punc), " ", essey1).lower()
essey2 = re.sub(r"[{}]+".format(punc), " ", essey2).lower()

print(type(essey1))


def get_dic_and_bow(essey:str):
    bow = essey.split()
    bow = [i for i in bow if i]
    wordDict = {}
    for word in bow:
        if word in wordDict:
            wordDict[word] += 1
        else:
            wordDict[word] = 1
    return wordDict, bow

def get_doc_list(esseys: List[str]):
    docList = []
    for idx, essey in enumerate(esseys):
        doc = {}
        bow = essey.split()
        bow = np.unique([i for i in bow if i])
        for idx_inner, word in enumerate(bow):
            if word in bow:
                if word in doc:
                    doc[word] += 1
                else:
                    doc[word] = 1



