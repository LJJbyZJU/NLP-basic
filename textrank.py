import spacy
import pytextrank

# 加载英文模型
nlp = spacy.load('en_core_web_sm')

# 添加TextRank到spaCy的管道
nlp.add_pipe("textrank")

# 处理文本
doc = nlp("Your preprocessed English text goes here.")

# 提取关键词
for phrase in doc._.phrases:
    print(phrase.text, phrase.rank)
