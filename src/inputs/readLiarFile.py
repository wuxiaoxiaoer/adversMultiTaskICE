import os
import nltk
from src.inputs.readFileUtil import *
from gensim.models import word2vec

def getLiarArticLabels(dir):
    file = []
    file = readFile(dir)
    en_stop_words = open('../commonData/en_stop_words.txt', encoding='utf-8')
    model = word2vec.Word2Vec.load("../commonData/text8.model")
    topics_list = []

    article_list = []
    labels = []
    credData = []

    for line in file:
        i = 0
        topic_comm_state_line = []
        cred = [0.5] * 200
        for col in line:
            col = col.strip().lower()
            if col == '':
                i += 1
                continue
            if i == 2:
                tokens = nltk.word_tokenize(col)
                clear_sw_tokens = []
                for word in tokens:
                    if word not in en_stop_words:
                        clear_sw_tokens.append(word)
                article_list.append(clear_sw_tokens)
            if i == 8:
                cred[0] = int(col)
            if i == 9:
                cred[1] = int(col)
            if i == 10:
                cred[2] = int(col)
            if i == 11:
                cred[3] = int(col)
            if i == 12:
                cred[4] = int(col)
            if i == 1:
                label = []
                if col == 'barely-true':
                    label = [[1, 0, 0, 0, 0, 0]]
                if col == 'false':
                    label = [[0, 1, 0, 0, 0, 0]]
                if col == 'half-true':
                    label = [[0, 0, 1, 0, 0, 0]]
                if col == 'mostly-true':
                    label = [[0, 0, 0, 1, 0, 0]]
                if col == 'pants-fire':
                    label = [[0, 0, 0, 0, 1, 0]]
                if col == 'true':
                    label = [[0, 0, 0, 0, 0, 1]]
                labels.append(label)
            i += 1
        credData.append(cred)

    article_vec = []
    for single in article_list:
        single_vec = []
        i = 0
        for word in single:
            if i < 30:
                word_vec = []
                try:
                    word_vec = model[word].tolist()
                except KeyError:
                    word_vec = [0.0]*200
                    pass
                single_vec.append(word_vec)
            i += 1
        for fill0 in range(len(single), 30):
            word_vec = [0.0] * 200
            single_vec.append(word_vec)
            pass
        article_vec.append(single_vec)
    return article_vec, credData, labels

# test
# article_vec, creData, labels = getLiarArticLabels(dir='../../data/liar_train_std.tsv')
# print(labels)

