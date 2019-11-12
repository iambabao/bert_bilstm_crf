import tensorflow as tf

from src.bert import BertConfig, BertModel


class BERTNER:
    def __init__(self, config, is_training):
        self.num_pos = config.num_pos
        self.num_tag = config.num_tag

        self.hidden_size = config.hidden_size

        self.lr = config.lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer
        self.is_training = is_training

        self.input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, None], name='segment_ids')
        self.input_length = tf.placeholder(tf.int32, [None], name='input_length')
        self.pos_ids = tf.placeholder(tf.int32, [None, None], name='pos_ids')
        self.tag_ids = tf.placeholder(tf.int32, [None, None], name='tag_ids')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.bert = BertModel(
            config=BertConfig.from_json_file(config.bert_config),
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False  # True if use TPU else False
        )
        self.pos_embedding = tf.keras.layers.Embedding(self.num_pos, 50)
        self.pos_dropout = tf.keras.layers.Dropout(self.dropout)
        self.encoder_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.encoder_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.final_layer = tf.keras.layers.Dense(self.num_tag)
        self.transition_params = tf.get_variable(
            shape=[self.num_tag, self.num_tag],
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            name='transition_params'
        )

        if config.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif config.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            assert False

        log_likelihood, self.pred_ids = self.forward()
        self.loss = -tf.reduce_mean(log_likelihood)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.tag_ids, self.pred_ids), tf.float32))
        self.gradients, self.train_op = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary = tf.summary.merge_all()

    def forward(self):
        # shape = [batch_size, seq_len, hidden_size + 50]
        input_em = self.embedding_layer()

        # shape = [batch_size, seq_len, 2 * hidden_size]
        input_enc = self.encoding_layer(input_em)

        # update transition_params with crf layer
        logits, log_likelihood, self.transition_params = self.crf_layer(input_enc)

        pred_ids = self.decoding_layer(logits)

        return log_likelihood, pred_ids

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def embedding_layer(self):
        context_em = self.bert.get_sequence_output()
        pos_em = self.pos_embedding(self.pos_ids)
        if self.is_training:
            pos_em = self.pos_dropout(pos_em)

        input_em = tf.concat([context_em, pos_em], axis=-1)

        return input_em

    def encoding_layer(self, input_em):
        with tf.variable_scope('encoder'):
            enc_output, _ = tf.nn.bidirectional_dynamic_rnn(
                self.encoder_cell_fw,
                self.encoder_cell_bw,
                input_em,
                sequence_length=self.input_length,
                dtype=tf.float32
            )
        enc_output = tf.concat(enc_output, axis=-1)

        return enc_output

    def crf_layer(self, input_enc):
        logits = self.final_layer(input_enc)

        # log_likelihood = [batch_size]
        # transition_params = [num_tag, num_tag]
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=self.tag_ids,
            sequence_lengths=self.input_length,
            transition_params=self.transition_params
        )

        return logits, log_likelihood, transition_params

    def decoding_layer(self, logits):
        pred_ids, _ = tf.contrib.crf.crf_decode(
            potentials=logits,
            transition_params=self.transition_params,
            sequence_length=self.input_length
        )

        return pred_ids
