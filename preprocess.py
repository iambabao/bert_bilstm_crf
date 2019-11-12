import json
import random
import collections

from src.config import Config
from src.utils import cut_text, pos_text, find_sub_span


def generate_data(config):
    data = []
    with open(config.raw_data, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = json.loads(line)

            context = []
            pos_seq = []
            for word, pos in pos_text(cut_text(line['context'])):
                context.append(word)
                pos_seq.append(pos)
            tag_seq = ['O'] * len(context)

            for key in line['argument'].keys():
                for value in line['argument'][key]:
                    value = cut_text(value)
                    start_idx = 0
                    while True:
                        start_idx = find_sub_span(context, value, start_idx)
                        if start_idx == -1:
                            break
                        # dealing with overlap between value
                        if tag_seq[start_idx] == 'O' and tag_seq[start_idx+len(value)-1] == 'O':
                            tag_seq[start_idx] = key.upper() + '-B'
                            for i in range(1, len(value)):
                                tag_seq[start_idx+i] = key.upper() + '-I'
                        start_idx += len(value)

            item = {'context': context, 'pos_seq': pos_seq, 'tag_seq': tag_seq}
            data.append(item)

            print('\rprocessing: {}'.format(len(data)), end='')
        print()

    random.shuffle(data)
    train_data_size = int(0.8 * len(data))
    train_data = data[:train_data_size]
    valid_data = data[train_data_size:]
    test_data = data[train_data_size:]

    with open(config.train_data, 'w', encoding='utf-8') as fout:
        for line in train_data:
            print(json.dumps(line, ensure_ascii=False), file=fout)
    with open(config.valid_data, 'w', encoding='utf-8') as fout:
        for line in valid_data:
            print(json.dumps(line, ensure_ascii=False), file=fout)
    with open(config.test_data, 'w', encoding='utf-8') as fout:
        for line in test_data:
            print(json.dumps(line, ensure_ascii=False), file=fout)


def build_dict(config):
    counter = collections.Counter()
    with open(config.train_data, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = json.loads(line)

            for pos in line['pos_seq']:
                counter[pos] += 1

    counter[config.pad] = 1e9 - config.pad_id
    counter[config.unk] = 1e9 - config.unk_id
    print('number of pos: {}'.format(len(counter)))

    pos_dict = {}
    for pos, _ in counter.most_common():
        pos_dict[pos] = len(pos_dict)

    with open(config.pos_dict, 'w', encoding='utf-8') as fout:
        json.dump(pos_dict, fout, ensure_ascii=False, indent=4)


def preprocess():
    config = Config('.', 'temp')

    print('generating data...')
    generate_data(config)

    print('building dict...')
    build_dict(config)

    print('done')


if __name__ == '__main__':
    preprocess()
