# __*__ coding:utf-8 __*__

import jieba
import jieba.analyse
from gensim.models import Word2Vec


def get_content():
    f = open("movie_info", 'r', encoding="utf-8")
    comment = []
    for lines in f.readlines():
        comment_data = []
        lines = lines.strip()
        lines = lines.split('\t')
        comment_data.append(lines[-2])
        comment.append(comment_data)
    return comment


def get_segment(content):
    segment = jieba.cut(content)
    keywords = jieba.analyse.extract_tags(content, topK=20)
    return keywords


def get_vector(keywords):
    model = Word2Vec(keywords, sg=1, size=100, window=5, min_count=2, negative=3, sample=0.001, hs=1)
    print(model.initialize_word_vectors())


comment = get_content()
for lines in comment:
    keywords = get_segment(lines[0])
    print(keywords)
    get_vector(keywords)
