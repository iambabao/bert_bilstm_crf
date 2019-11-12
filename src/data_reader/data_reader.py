import json

from src.utils import convert_list, cut_text, pos_text


def _convert_single_example(inputs, pos_seq, tag_seq, max_seq_length, tokenizer):
    input_tokens = []
    pos_tokens = []
    tag_tokens = []
    temp = [tokenizer.tokenize(inp) for inp in inputs]
    for i, (pos, tag) in enumerate(zip(pos_seq, tag_seq)):
        input_tokens += temp[i]
        pos_tokens += [pos] * len(temp[i])
        if tag.endswith('B'):
            tag_tokens += [tag] + [tag[:-1] + 'I'] * (len(temp[i]) - 1)
        else:
            tag_tokens += [tag] * len(temp[i])

    # Account for [CLS] and [SEP] with "- 2"
    input_tokens = ['[CLS]'] + input_tokens[0:(max_seq_length - 2)] + ['[SEP]']
    pos_tokens = ['<pad>'] + pos_tokens[0:(max_seq_length - 2)] + ['<pad>']
    tag_tokens = ['O'] + tag_tokens[0:(max_seq_length - 2)] + ['O']
    input_length = len(input_tokens)

    assert len(pos_tokens) == input_length
    assert len(tag_tokens) == input_length

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    segment_ids = [0] * input_length

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * input_length

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        pos_tokens.append('<pad>')
        tag_tokens.append('O')

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(pos_tokens) == max_seq_length
    assert len(tag_tokens) == max_seq_length

    return input_ids, input_mask, segment_ids, input_length, pos_tokens, tag_tokens


class DataReader:
    def __init__(self, config, tokenizer, pos_2_id, tag_2_id):
        self.config = config
        self.tokenizer = tokenizer
        self.pos_2_id = pos_2_id
        self.tag_2_id = tag_2_id

    def _read_data(self, data_file):
        input_ids = []
        input_mask = []
        segment_ids = []
        input_length = []
        pos_ids = []
        tag_ids = []

        counter = 0
        with open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = json.loads(line)
                context = line['context']
                pos_seq = line['pos_seq']
                tag_seq = line['tag_seq']

                v1, v2, v3, v4, v5, v6 = _convert_single_example(
                    context, pos_seq, tag_seq, self.config.sequence_len, self.tokenizer
                )
                v5 = convert_list(v5, self.pos_2_id, self.config.pad_id, self.config.unk_id)
                v6 = convert_list(v6, self.tag_2_id, 0, 0)

                input_ids.append(v1)
                input_mask.append(v2)
                segment_ids.append(v3)
                input_length.append(v4)
                pos_ids.append(v5)
                tag_ids.append(v6)

                counter += 1
                print('\rprocessing: {}'.format(counter), end='')
            print()

        return input_ids, input_mask, segment_ids, input_length, pos_ids, tag_ids

    def read_train_data(self):
        return self._read_data(self.config.train_data)

    def read_valid_data(self):
        return self._read_data(self.config.valid_data)

    def read_test_data(self):
        return self._read_data(self.config.test_data)

    def convert_data(self, context):
        context_seq = []
        pos_seq = []
        for word, pos in pos_text(cut_text(context)):
            context_seq.append(word)
            pos_seq.append(pos)

        input_tokens = []
        pos_tokens = []
        temp = [self.tokenizer.tokenize(word) for word in context_seq]
        for i, pos in enumerate(pos_seq):
            input_tokens += temp[i]
            pos_tokens += [pos] * len(temp[i])

        # Account for [CLS] and [SEP] with "- 2"
        input_tokens = ['[CLS]'] + input_tokens[0:(self.config.sequence_len - 2)] + ['[SEP]']
        pos_tokens = ['<pad>'] + pos_tokens[0:(self.config.sequence_len - 2)] + ['<pad>']
        input_length = len(input_tokens)

        assert len(pos_tokens) == input_length

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0] * input_length
        input_mask = [1] * input_length
        pos_ids = convert_list(pos_tokens, self.pos_2_id, self.config.pad_id, self.config.unk_id)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.config.sequence_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            pos_ids.append(self.config.pad_id)

        assert len(input_ids) == self.config.sequence_len
        assert len(input_mask) == self.config.sequence_len
        assert len(segment_ids) == self.config.sequence_len
        assert len(pos_ids) == self.config.sequence_len

        return input_ids, input_mask, segment_ids, input_length, pos_ids

    def load_data_from_file(self, data_file):
        input_ids = []
        input_mask = []
        segment_ids = []
        input_length = []
        pos_ids = []

        counter = 0
        with open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                v1, v2, v3, v4, v5 = self.convert_data(line.strip())
                input_ids.append(v1)
                input_mask.append(v2)
                segment_ids.append(v3)
                input_length.append(v4)
                pos_ids.append(v5)

                counter += 1
                print('\rprocessing: {}'.format(counter), end='')
            print()

        return input_ids, input_mask, segment_ids, input_length, pos_ids
