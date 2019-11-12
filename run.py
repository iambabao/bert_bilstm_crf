import os
import time
import json
import argparse
import tensorflow as tf

from src.bert import FullTokenizer, get_assignment_map_from_checkpoint
from src.config import Config
from src.data_reader import DataReader
from src.model import get_model
from src.utils import read_dict, make_batch_iter, convert_list, parse_output

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--em', action='store_true', default=False)
args = parser.parse_args()

config = Config('.', args.model, num_epoch=args.epoch, batch_size=args.batch,
                optimizer=args.optimizer, lr=args.lr,
                embedding_trainable=args.em)

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


def evaluate(sess, model, batch_iter, verbose=True):
    steps = 0
    total_loss = 0.0
    total_accuracy = 0.0
    outputs = []
    for batch in batch_iter:
        input_ids, input_mask, segment_ids, input_length, pos_ids, tag_ids = list(zip(*batch))

        pred_ids, loss, accuracy = sess.run(
            [model.pred_ids, model.loss, model.accuracy],
            feed_dict={
                model.input_ids: input_ids,
                model.input_mask: input_mask,
                model.segment_ids: segment_ids,
                model.input_length: input_length,
                model.pos_ids: pos_ids,
                model.tag_ids: tag_ids
            }
        )
        outputs.extend(refine_output(input_ids, pred_ids, input_length))

        steps += 1
        total_loss += loss
        total_accuracy += accuracy
        if verbose:
            print('\rprocessing batch: {:>6d}'.format(steps + 1), end='')
    print()

    return outputs, total_loss / steps, total_accuracy / steps


def run_epoch(sess, model, batch_iter, summary_writer, verbose=True):
    steps = 0
    total_loss = 0.0
    total_accuracy = 0.0
    start_time = time.time()
    for batch in batch_iter:
        input_ids, input_mask, segment_ids, input_length, pos_ids, tag_ids = list(zip(*batch))

        _, loss, accuracy, global_step, summary = sess.run(
            [model.train_op, model.loss, model.accuracy, model.global_step, model.summary],
            feed_dict={
                model.input_ids: input_ids,
                model.input_mask: input_mask,
                model.segment_ids: segment_ids,
                model.input_length: input_length,
                model.pos_ids: pos_ids,
                model.tag_ids: tag_ids
            }
        )

        steps += 1
        total_loss += loss
        total_accuracy += accuracy
        if verbose and steps % 10 == 0:
            summary_writer.add_summary(summary, global_step)

            current_time = time.time()
            time_per_batch = (current_time - start_time) / 10.0
            start_time = current_time
            print('\rAfter {:>6d} batch(es), loss is {:>.4f}, accuracy is {:>.4f}, {:>.4f}s/batch'.format(
                steps, loss, accuracy, time_per_batch), end='')
    print()

    return total_loss / steps, total_accuracy / steps


def train():
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.train_log_dir):
        os.mkdir(config.train_log_dir)
    if not os.path.exists(config.valid_log_dir):
        os.mkdir(config.valid_log_dir)

    print('loading data...')
    tokenizer = FullTokenizer(config.bert_vocab, do_lower_case=config.to_lower)
    pos_2_id, id_2_pos = read_dict(config.pos_dict)
    tag_2_id, id_2_tag = read_dict(config.tag_dict)
    config.num_pos = len(pos_2_id)
    config.num_tag = len(tag_2_id)

    data_reader = DataReader(config, tokenizer, pos_2_id, tag_2_id)
    train_data = data_reader.read_train_data()
    valid_data = data_reader.read_valid_data()

    print('building model...')
    model = get_model(config, is_training=True)

    tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, config.bert_ckpt)
    tf.train.init_from_checkpoint(config.bert_ckpt, assignment_map)

    print('==========  Trainable Variables  ==========')
    for v in tvars:
        init_string = ''
        if v.name in initialized_variable_names:
            init_string = '<INIT_FROM_CKPT>'
        print(v.name, v.shape, init_string)

    print('==========  Gradients  ==========')
    for g in model.gradients:
        print(g)

    best_score = 0.0
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=sess_config) as sess:
        if tf.train.latest_checkpoint(config.result_dir):
            saver.restore(sess, tf.train.latest_checkpoint(config.result_dir))
            print('loading model from {}'.format(tf.train.latest_checkpoint(config.result_dir)))
        else:
            tf.global_variables_initializer().run()
            print('initializing from scratch.')

        train_writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)

        for i in range(config.num_epoch):
            print('==========  Epoch {} Train  =========='.format(i + 1))
            train_batch_iter = make_batch_iter(list(zip(*train_data)), config.batch_size, shuffle=True)
            train_loss, train_accu = run_epoch(sess, model, train_batch_iter, train_writer, verbose=True)
            print('The average train loss is {:>.4f}, average train accuracy is {:>.4f}'.format(train_loss, train_accu))

            print('==========  Epoch {} Valid  =========='.format(i + 1))
            valid_batch_iter = make_batch_iter(list(zip(*valid_data)), config.batch_size, shuffle=False)
            outputs, valid_loss, valid_accu = evaluate(sess, model, valid_batch_iter, verbose=True)
            print('The average valid loss is {:>.4f}, average valid accuracy is {:>.4f}'.format(valid_loss, valid_accu))

            print('==========  Saving Result  ==========')
            save_result(outputs, config.valid_result, tokenizer, id_2_tag)

            if valid_accu > best_score:
                best_score = valid_accu
                saver.save(sess, config.model_file)


def test():
    print('loading data...')
    tokenizer = FullTokenizer(config.bert_vocab, do_lower_case=config.to_lower)
    pos_2_id, id_2_pos = read_dict(config.pos_dict)
    tag_2_id, id_2_tag = read_dict(config.tag_dict)
    config.num_pos = len(pos_2_id)
    config.num_tag = len(tag_2_id)

    data_reader = DataReader(config, tokenizer, pos_2_id, tag_2_id)
    test_data = data_reader.read_test_data()

    print('building model...')
    model = get_model(config, is_training=False)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=sess_config) as sess:
        if tf.train.latest_checkpoint(config.result_dir):
            saver.restore(sess, tf.train.latest_checkpoint(config.result_dir))
            print('loading model from {}'.format(tf.train.latest_checkpoint(config.result_dir)))

            print('==========  Test  ==========')
            test_batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False)
            outputs, test_loss, test_accu = evaluate(sess, model, test_batch_iter, verbose=True)
            print('The average test loss is {:>.4f}, average test accuracy is {:>.4f}'.format(test_loss, test_accu))

            print('==========  Saving Result  ==========')
            save_result(outputs, config.test_result, tokenizer, id_2_tag)
        else:
            print('model not found.')

        print('done')


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        assert False
