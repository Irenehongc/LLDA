# __*__ coding:utf-8 __*__
import re

import jieba
import jieba.analyse
import multiprocessing
from gensim.models import Word2Vec


def get_content(punctuation):
    """
    切词得到训练语料库train_corpus.txt
    """
    with open("movie_info", 'r', encoding="utf-8") as f, open("train_corpus.txt", 'w', encoding="utf-8") as w:
        for lines in f.readlines():
            lines = lines.strip()
            movie_comment = lines.split('\t')
            info = re.sub('[{0}]+'.format(punctuation), '', movie_comment[-2])
            words = jieba.cut(info)
            w.write(' '.join(words))


def get_w2vector():
    train_corpus_text = 'train_corpus.txt'
    # 是每个词的向量维度
    size = 400
    # 是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词
    window = 5
    # 设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃
    min_count = 1
    # 是训练的进程数，默认是当前运行机器的处理器核数。
    workers = multiprocessing.cpu_count()
    # w2v模型文件
    model_text = 'w2v_size_{0}.model'.format(size)
    # w2v训练模型
    sentences = word2vec.Text8Corpus(train_corpus_text)
    print(sentences)
    model = word2vec.Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=workers)
    model.save(model_text)
    return model


# 严格限制标点符号
strict_punctuation = '。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
# 简单限制标点符号
simple_punctuation = '’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# 去除标点符号
punctuation = simple_punctuation + strict_punctuation
get_content(punctuation)
model = get_w2vector()
