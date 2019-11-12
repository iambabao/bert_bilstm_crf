import json
import numpy as np
import math
import random
import jieba
import nltk
import joblib
import tensorflow as tf
from jieba import posseg as pseg
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def pad_list(item_list, pad, max_len):
    item_list = item_list[:max_len]
    return item_list + [pad] * (max_len - len(item_list))


def pad_batch(data_batch, pad, max_len=None):
    if max_len is None:
        max_len = len(max(data_batch, key=len))
    return [pad_list(data, pad, max_len) for data in data_batch]


def convert_item(item, convert_dict, unk):
    return convert_dict[item] if item in convert_dict else unk


def convert_list(item_list, convert_dict, pad, unk, max_len=None):
    item_list = [convert_item(item, convert_dict, unk) for item in item_list]
    if max_len is not None:
        item_list = pad_list(item_list, pad, max_len)

    return item_list


def read_dict(dict_file):
    with open(dict_file, 'r', encoding='utf-8') as fin:
        key_2_id = json.load(fin)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def cut_text(text, language='english'):
    text = text.strip()
    return nltk.word_tokenize(text, language=language)


def cut_text_zh(text):
    text = text.strip()
    return jieba.lcut(text)


def pos_text(words, lang='eng'):
    return nltk.pos_tag(words, lang=lang)


def pos_text_zh(text):
    text = text.strip()
    return pseg.lcut(text)


def make_batch_iter(data, batch_size, shuffle):
    data_size = len(data)

    if shuffle:
        random.shuffle(data)

    num_batches = (data_size + batch_size - 1) // batch_size
    print('total batches: {}'.format(num_batches))
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(data_size, (i + 1) * batch_size)
        yield data[start_index: end_index]


def train_embedding(text_file, embedding_size, model_file):
    data = []
    with open(text_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(line.strip().split())

    model = Word2Vec(data, size=embedding_size, window=5, min_count=5, workers=8)
    model.save(model_file)


def load_embedding(model_file, word_list):
    model = Word2Vec.load(model_file)

    embedding_matrix = []
    for word in word_list:
        if word in model:
            embedding_matrix.append(model[word])
        else:
            embedding_matrix.append(np.zeros(model.vector_size))

    return np.array(embedding_matrix)


def load_glove_embedding(data_file, word_list):
    w2v = {}
    with open(data_file, 'r', encoding='utf-8') as fin:
        line = fin.readline()
        embedding_size = len(line.strip().split()) - 1
        while line is not None:
            line = line.strip().split()
            word = line[0]
            vector = [float(val) for val in line[1:]]
            if word in word_list:
                w2v[word] = vector
            line = fin.readline()
    print('hit words: {}'.format(len(w2v)))

    embedding_matrix = []
    for word in word_list:
        if word in w2v:
            embedding_matrix.append(w2v[word])
        else:
            embedding_matrix.append([0.0] * embedding_size)
    return np.array(embedding_matrix), embedding_size


def train_tfidf(text_file, feature_size, model_file):
    with open(text_file, 'r', encoding='utf-8') as fin:
        data = fin.readlines()

    tfidf = TfidfVectorizer(
        max_features=feature_size,
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b\w+\b'
    ).fit(data)

    joblib.dump(tfidf, model_file)


def load_tfidf(model_file):
    return joblib.load(model_file)


def load_gidf(model_file):
    gidf = {}
    with open(model_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            w, v = line.strip().split()
            gidf[w] = float(v)

    return gidf


def cosine_similarity(v1, v2):
    r = 0
    s1 = 0.0
    s2 = 0.0
    for x, y in zip(v1, v2):
        r += x * y
        s1 += x * x
        s2 += y * y
    return r / (math.sqrt(s1) * math.sqrt(s2) + 1e-10)


def view_tf_check_point(ckpt_dir_or_file):
    init_vars = tf.train.list_variables(ckpt_dir_or_file)

    for v in init_vars:
        print(v)


# ====================
def find_sub_span(context, span, start=0):
    span_len = len(span)
    for i in range(start, len(context) - span_len + 1):
        if context[i:i + span_len] == span:
            return i

    return -1


def parse_output(output, context):
    result = {'argument': {}, 'context': parse_context(context[1:-1])}
    arg_type = ''
    arg_start = 0
    for i, v in enumerate(output):
        if v != 'O':
            t1, t2 = v.split('-')
            if arg_type == '':
                if t2 == 'B':
                    arg_type = t1
                    arg_start = i
            elif not (t1 == arg_type and t2 == 'I'):
                result['argument'][arg_type.lower()] = result['argument'].get(arg_type.lower(), []) + [parse_context(context[arg_start:i])]
                if t2 == 'B':
                    arg_type = t1
                    arg_start = i
                else:
                    arg_type = ''
        elif arg_type != '':
            result['argument'][arg_type.lower()] = result['argument'].get(arg_type.lower(), []) + [parse_context(context[arg_start:i])]
            arg_type = ''

    for key in result['argument'].keys():
        result['argument'][key] = list(set(result['argument'][key]))

    return result


def parse_context(context):
    parent_index = 0
    for i, word in enumerate(context):
        if word.startswith('##'):
            context[parent_index] += word[2:]
            context[i] = ''
        else:
            parent_index = i
    
    context = ' '.join([word for word in context if word != ''])

    return context
