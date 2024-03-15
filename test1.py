import re
import math
import matplotlib.pyplot as plt

from string import punctuation as punc
from collections import Counter
from wordcloud import WordCloud

# with open("./example/51120", "r") as f:
#     essey1 = f.read()
# with open("./example/51121", "r") as f:
#     essey2 = f.read()

with open("./20news-18828/alt.atheism/51060", "r") as f:
    essey1 = f.read()
with open("./example/51121", "r") as f:
    essey2 = f.read()

punc += "\n"

# essey1和essey2经过处理以后，将文章中的单词使用空格隔开，且大写变为小写，方便后续处理
essey1 = re.sub(r"[{}]+".format(punc), " ", essey1).lower()
essey2 = re.sub(r"[{}]+".format(punc), " ", essey2).lower()

# 生成词语列表
word_list1 = essey1.split()
word_list2 = essey2.split()

# 读取停用词文件
with open('stopwords.txt', 'r') as file:
    stopwords = file.read().splitlines()
# 过滤停用词
filtered_words_list1 = [word for word in word_list1 if word not in stopwords]
filtered_words_list2 = [word for word in word_list2 if word not in stopwords]

# 生成词语字典
word_dict1 = Counter(word_list1)
word_dict2 = Counter(word_list2)

# 验证结果
# print(word_list1)
# print(word_list2)
# print(word_dict1)
# print(word_dict2)

# 计算文档1的TF
total_words1 = sum(word_dict1.values())
tf1 = {word: count / total_words1 for word, count in word_dict1.items()}

# 计算文档2的TF
total_words2 = sum(word_dict2.values())
tf2 = {word: count / total_words2 for word, count in word_dict2.items()}

# 文档总数
num_docs = 2
# 包含每个词的文档数量
doc_count = {word: sum(1 for doc_words in [word_dict1, word_dict2] if word in doc_words) for word in set(word_dict1) | set(word_dict2)}

# 计算IDF
idf = {word: math.log(num_docs / (count + 1)) for word, count in doc_count.items()}

# 计算文档1的TF-IDF
tfidf1 = {word: tf * idf[word] for word, tf in tf1.items()}

# 计算文档2的TF-IDF
tfidf2 = {word: tf * idf[word] for word, tf in tf2.items()}

# 验证结果
# print('tf1:', tf1)
# print('tf2:', tf2)
# print('idf:', idf)
# print('tfidf1:', tfidf1)
# print('tfidf2:', tfidf2)

# 提取文章1的前10个关键词
keywords1 = sorted(tfidf1.items(), key=lambda item: item[1], reverse=True)[:10]

# 提取文章2的前10个关键词
keywords2 = sorted(tfidf2.items(), key=lambda item: item[1], reverse=True)[:10]

# 打印关键词
print("文章1的关键词：", [word for word, tfidf in keywords1])
print("文章2的关键词：", [word for word, tfidf in keywords2])

# 转换为字典
dict1 = {key: value for key, value in keywords1}
dict2 = {key: value for key, value in keywords2}
print("文档1的关键词字典:", dict1)
print("文档2的关键词字典:", dict2)

# 假设keywords是您从文章中提取出的关键词列表
keywords = {'keyword1': 3.6, 'keyword2': 2.5, 'keyword3': 1.8, 'keyword4': 1.2, 'keyword5': 1.0, 'keyword6': 0.9, 'keyword7': 0.8, 'keyword8': 0.7, 'keyword9': 0.6, 'keyword10': 0.5}

# 生成词云
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)

# 显示词云图
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



