import os
import re
from math import log
import matplotlib.pyplot as plt

from string import punctuation as punc
from collections import Counter
from wordcloud import WordCloud

# 关键词列表
results = {}

# 设置文件夹路径
# folder_path = './20news-18828/comp.os.ms-windows.misc'
# root_folder_path = './20news-18828'
root_folder_path = './example'

# 获取文件夹中所有文件的名称
# files = os.listdir(folder_path)
# print(files)

punc += "\n"
doc_frequency = {}

# 读取停用词文件
with open('stopwords.txt', 'r') as file:
    stopwords = file.read().splitlines()


# def pretreatment(root_path):
#     global doc_frequency
#     # 遍历文件夹中的每个文件
#     for folder_name, subfolders, files in os.walk(root_path):
#         for file in files:
#             # # 构造文件的绝对路径
#             # file_path = os.path.join(folder_path, file)
#             # # file_path = './20news-18828/comp.graphics/37913'
#             # print(file_path)
#             # 构造文件的绝对路径
#             file_path = os.path.join(folder_name, file)
#
#             with open(file_path, 'r') as f:
#                 data = f.read()
#
#             data = re.sub(r"[{}]+".format(punc), " ", data).lower()
#             word_list = data.split()
#             filtered_words_list = [word for word in word_list if word not in stopwords]
#             word_dict = Counter(filtered_words_list)
#
#             # 更新文档频率DF
#             for word in word_dict:
#                 doc_frequency[word] = doc_frequency.get(word, 0) + 1
#
#             # 计算TF
#             total_words = sum(word_dict.values())
#             tf = {word: count / total_words for word, count in word_dict.items()}
#
#             # 存储TF值
#             results[file] = {'tf': tf}

def pretreatment(root_path):
    global doc_frequency
    # 遍历文件夹中的每个文件
    for folder_name, subfolders, files in os.walk(root_path):
        # 初始化当前文件夹的结果字典
        folder_results = {}
        for file in files:
            # 构造文件的绝对路径
            file_path = os.path.join(folder_name, file)

            with open(file_path, 'r') as f:
                data = f.read()

            # 数据预处理
            data = re.sub(r"[{}]+".format(re.escape(punc)), " ", data).lower()
            word_list = data.split()
            filtered_words_list = [word for word in word_list if word not in stopwords]
            word_dict = Counter(filtered_words_list)

            # 更新文档频率DF
            for word in word_dict:
                doc_frequency[word] = doc_frequency.get(word, 0) + 1

            # 计算TF
            total_words = sum(word_dict.values())
            tf = {word: count / total_words for word, count in word_dict.items()}

            # 存储TF值到当前文件夹的结果字典
            folder_results[file] = {'tf': tf}

        # 将当前文件夹的结果存储到主结果字典
        results[folder_name] = folder_results


# 调用函数处理所有文件
pretreatment(root_folder_path)
# 打印层次化的结果
for folder, folder_results in results.items():
    print(f"文件夹: {folder}")
    for file, values in folder_results.items():
        print(f"  文件: {file}, TF值: {values['tf']}")

# # 计算IDF
# num_docs = len(results)
# idf = {word: log(num_docs / df) for word, df in doc_frequency.items()}
#
# # 计算TF-IDF并写入results
# for file, values in results.items():
#     tf = values['tf']
#     tf_idf = {word: tf[word] * idf[word] for word in tf}
#     # results[file]['tf-idf'] = tf_idf
#     # 对TF-IDF字典进行排序，选择值最高的前10个词
#     top_10_keywords = sorted(tf_idf.items(), key=lambda item: item[1], reverse=True)[:10]
#     # 将排序后的前10个关键词存储回results字典
#     results[file]['top-10-tf-idf'] = top_10_keywords
#
# # 所有词
# # for file, values in results.items():
# #     print(f"文件: {file}, TF-IDF值: {values['tf-idf']}")
#
# # 前10个关键词
# for file, values in results.items():
#     print(f"文件: {file}, 前10个关键词: {values['top-10-tf-idf']}")
#
# first_file = results[0]
# # first_file_tf_idf = results[first_file]['tf-idf']
# first_file_tf_idf = results[first_file]['top-10-tf-idf']
# # 使用字典推导式将列表转换为字典
# keyword_dict = {key: value for key, value in first_file_tf_idf}
# print(keyword_dict)
#
#
# # 生成词云
# def cloud_word(dict):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict)
#
#     # 显示词云图
#     plt.figure(figsize=(15, 7))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.show()
#
#
# # cloud_word(keyword_dict)
