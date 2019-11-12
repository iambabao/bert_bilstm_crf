import os
import json
import argparse
import tensorflow as tf

from src.bert import FullTokenizer
from src.config import Config
from src.data_reader import DataReader
from src.model import get_model
from src.utils import read_dict, make_batch_iter, convert_list, parse_output

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--input', '-i', type=str, required=True)
parser.add_argument('--output', '-o', type=str, required=True)
parser.add_argument('--batch', type=int, default=32)
args = parser.parse_args()

config = Config('.', args.model, batch_size=args.batch)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def save_result(outputs, result_file, tokenizer, id_2_tag):
    print('write file: {}'.format(result_file))
    with open(result_file, 'w', encoding='utf-8') as fout:
        for context, tags in outputs:
            context = tokenizer.convert_ids_to_tokens(context)
            tags = convert_list(tags, id_2_tag, 'O', 'O')
            result = parse_output(tags, context)
            print(json.dumps(result, ensure_ascii=False), file=fout)


def refine_output(input_ids, pred_ids, input_length):
    outputs = []
    for x, y, z in zip(input_ids, pred_ids, input_length):
        outputs.append((x[:z], y[:z]))

    return outputs


def inference(sess, model, batch_iter, verbose=True):
    steps = 0
    outputs = []
    for batch in batch_iter:
        input_ids, input_mask, segment_ids, input_length, pos_ids = list(zip(*batch))

        pred_ids = sess.run(
            model.pred_ids,
            feed_dict={
                model.input_ids: input_ids,
                model.input_mask: input_mask,
                model.segment_ids: segment_ids,
                model.input_length: input_length,
                model.pos_ids: pos_ids
            }
        )

        steps += 1
        outputs.extend(refine_output(input_ids, pred_ids, input_length))
        if verbose:
            print('\rprocessing batch: {:>6d}'.format(steps + 1), end='')
    print()

    return outputs


def main():
    print('loading data...')
    tokenizer = FullTokenizer(config.bert_vocab, do_lower_case=config.to_lower)
    pos_2_id, id_2_pos = read_dict(config.pos_dict)
    tag_2_id, id_2_tag = read_dict(config.tag_dict)
    config.num_pos = len(pos_2_id)
    config.num_tag = len(tag_2_id)

    data_reader = DataReader(config, tokenizer, pos_2_id, tag_2_id)
    input_file = args.input
    print('input file: {}'.format(input_file))
    input_data = data_reader.load_data_from_file(input_file)    

    print('building model...')
    model = get_model(config, is_training=False)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=sess_config) as sess:
        if tf.train.latest_checkpoint(config.result_dir):
            saver.restore(sess, tf.train.latest_checkpoint(config.result_dir))
            print('loading model from {}'.format(tf.train.latest_checkpoint(config.result_dir)))

            batch_iter = make_batch_iter(list(zip(*input_data)), config.batch_size, shuffle=False)
            outputs = inference(sess, model, batch_iter, verbose=True)

            print('==========  Saving Result  ==========')
            output_file = args.output
            save_result(outputs, output_file, tokenizer, id_2_tag)
        else:
            print('model not found.')

        print('done')


if __name__ == '__main__':
     main()
