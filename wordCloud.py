import json

from wordcloud import WordCloud
import matplotlib.pyplot as plt

json_filename = './results/alt.atheism/alt.atheism_tf_idf.json'

with open(json_filename, 'r', encoding='utf-8') as f:
    tf_idf_data = json.load(f)

first_article_filename = next(iter(tf_idf_data))  # 第一篇文章
article_filename = '49960'  # 指定任意文章
article_tf_idf = tf_idf_data[article_filename]['tf-idf']

# 生成词云图
wordcloud = WordCloud(width=800, height=400, background_color='white')
wordcloud.generate_from_frequencies(article_tf_idf)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

