import numpy as np

from src.bert import FullTokenizer as Tokenizer
from src.config import Config
from src.data_reader import DataReader
from src.utils import read_dict, convert_list, parse_output


def refine_output(input_ids, pred_ids, input_length, tokenizer, id_2_tag):
    context = tokenizer.convert_ids_to_tokens(input_ids)
    pred_tags = convert_list(pred_ids, id_2_tag, 'O', 'O')

    return parse_output(pred_tags[:input_length], context[:input_length])


def check_data(data, tokenizer, id_2_pos, id_2_tag):
    input_ids, input_mask, segment_ids, input_length, pos_ids, tag_ids = data

    for _ in range(5):
        print('=' * 20)
        index = np.random.randint(0, len(input_ids))
        print('id: {}'.format(index))
        length = input_length[index]

        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[index])
        print('input tokens: {}'.format(input_tokens[:length]))
        pos_tokens = convert_list(pos_ids[index], id_2_pos, '<pad>', '<unk>')
        print('pos tokens: {}'.format(pos_tokens[:length]))
        tag_tokens = convert_list(tag_ids[index], id_2_tag, 'O', 'O')
        print('tag tokens: {}'.format(tag_tokens[:length]))

        result = refine_output(input_ids[index], tag_ids[index], length, tokenizer, id_2_tag)
        print(result)


def main():
    config = Config('.', 'temp')
    pos_2_id, id_2_pos = read_dict(config.pos_dict)
    tag_2_id, id_2_tag = read_dict(config.tag_dict)
    tokenizer = Tokenizer(config.bert_vocab, do_lower_case=config.to_lower)
    data_reader = DataReader(config, tokenizer, pos_2_id, tag_2_id)

    valid_data = data_reader.read_valid_data()
    check_data(valid_data, tokenizer, id_2_pos, id_2_tag)

    print('done')


if __name__ == '__main__':
    main()
