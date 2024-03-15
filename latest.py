import os
import re
import json

from math import log
from string import punctuation as punc
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# TF
tf_results = {}

# 设置文件夹路径
root_folder_path = './example'

punc += "\n"
# 初始化文档频率字典
doc_frequency = {}

# 读取停用词文件
with open('./stopwords.txt', 'r') as file:
    stopwords = file.read().splitlines()

# 添加额外的标点符号到停用词列表
stopwords.extend("\n")


def pretreatment(root_path):
    global doc_frequency
    # 遍历文件
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
            # 过滤停用词和数字
            filtered_words_list = [word for word in word_list if word not in stopwords and not word.isdigit()]
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
        tf_results[folder_name] = folder_results


# 调用函数处理所有文件
pretreatment(root_folder_path)
# print(tf_results)

# # 打印层次化的结果
# for folder, folder_results in tf_results.items():
#     print(f"文件夹: {folder}")
#     for file, values in folder_results.items():
#         print(f"  文件: {file}, TF值: {values['tf']}")

# IDF
num_docs = sum([len(files) for _, _, files in os.walk(root_folder_path)])
idf_results = {word: log(num_docs / df) for word, df in doc_frequency.items()}

# TF-IDF
tf_idf_results = {}

# 计算TF-IDF并写入不同的字典中
for folder, folder_results in tf_results.items():
    tf_idf_folder_results = {}
    for file, values in folder_results.items():
        tf = values['tf']
        tf_idf = {word: tf[word] * idf_results[word] for word in tf}
        tf_idf_folder_results[file] = {'tf-idf': tf_idf}
    tf_idf_results[folder] = tf_idf_folder_results


# 写入一个JSON文件
# json_data = {}
# for folder, folder_results in tf_results.items():
#     folder_data = {}
#     for file, values in folder_results.items():
#         file_data = {
#             'tf': values['tf'],
#             'idf': {word: idf_results[word] for word in values['tf'].keys()},
#             'tf-idf': tf_idf_results[folder][file]['tf-idf']
#         }
#         folder_data[file] = file_data
#     json_data[folder] = folder_data
# with open('./results.json', 'w', encoding='utf-8') as json_file:
#     json.dump(json_data, json_file, ensure_ascii=False, indent=4)

# 遍历子文件夹
for folder, folder_results in tf_results.items():
    if folder == root_folder_path:
        continue  # 如果是根目录，则跳过
    # 创建包含TF、IDF和TF-IDF值的字典
    folder_data = {
        'tf': folder_results,
        'idf': idf_results,
        'tf-idf': tf_idf_results[folder]
    }

    # 为每个子文件夹创建JSON文件
    folder_name = os.path.basename(folder)
    folder_path = f'./results/{folder_name}'

    # 确保目录存在
    os.makedirs(folder_path, exist_ok=True)

    # 为每个子文件夹创建JSON文件
    folder_name = os.path.basename(folder)
    with open(f'{folder_path}/{folder_name}_tf.json', 'w', encoding='utf-8') as tf_file:
        json.dump(folder_data['tf'], tf_file, ensure_ascii=False, indent=4)
    with open(f'{folder_path}/{folder_name}_idf.json', 'w', encoding='utf-8') as idf_file:
        json.dump(folder_data['idf'], idf_file, ensure_ascii=False, indent=4)
    with open(f'{folder_path}/{folder_name}_tf_idf.json', 'w', encoding='utf-8') as tf_idf_file:
        json.dump(folder_data['tf-idf'], tf_idf_file, ensure_ascii=False, indent=4)

json_filename = './results/test1/test1_tf_idf.json'

with open(json_filename, 'r', encoding='utf-8') as f:
    tf_idf_data = json.load(f)

first_article_filename = next(iter(tf_idf_data))
article_tf_idf = tf_idf_data[first_article_filename]['tf-idf']

# 生成词云图
wordcloud = WordCloud(width=800, height=400, background_color='white')
wordcloud.generate_from_frequencies(article_tf_idf)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

