import tensorflow as tf
import numpy as np
class Seq2Seq:

    # 3개의 layer와 128개의 hidden_node 사용
    def __init__(self, input_max_len, output_max_len, vocab_size, word_embedding_matrix, training_mode, n_hidden=128, n_layers=3):
        self.learning_late = 0.001

        self.input_max_len = input_max_len
        self.output_max_len = output_max_len
        self.vocab_size = vocab_size
        self.word_embedding_matrix = word_embedding_matrix
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.training_mode = training_mode
        self.batch_size = 26

        self.enc_input_idx = tf.placeholder(tf.int32, [None, self.input_max_len])
        self.dec_input_idx = tf.placeholder(tf.int32, [None, self.output_max_len])
        self.targets = tf.placeholder(tf.int32, [None, None])   # [batct_size, words_len]
        self.enc_length = tf.placeholder(tf.int32, [None])
        self.dec_length = tf.placeholder(tf.int32, [None])

        # 학습 총 회수
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # 모델 구축
        self.build_model()

        # 학습회수 저장
        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        # cell 구축
        enc_cell, dec_cell, context_cell = self.build_cells()

        # encode, context, decode 구축
        with tf.variable_scope('encode', reuse=tf.AUTO_REUSE):
            input_sentence_emb = tf.nn.embedding_lookup(self.word_embedding_matrix, self.enc_input_idx)
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, input_sentence_emb, self.enc_length, dtype=tf.float32)
            # enc_states -> [batch_size, hidden node]

        # context_input -> [batch_size, 1, hidden node]
        with tf.variable_scope('context', reuse=tf.AUTO_REUSE):
            context_input = tf.reshape(enc_states, [-1, 1, self.n_hidden])
            outputs, context_states = tf.nn.dynamic_rnn(context_cell, context_input, dtype=tf.float32)

        outputs = tf.reshape(outputs, [-1, self.n_hidden])

        with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
            output_sentence_emb = tf.nn.embedding_lookup(self.word_embedding_matrix, self.dec_input_idx)
            # 학습 경우
            if self.training_mode is True:
                helper = tf.contrib.seq2seq.TrainingHelper(output_sentence_emb, self.dec_length)

            # 학습이 아닌 경우
            else:
                start_tokens = tf.fill([self.batch_size], 1)
                end_token = 2
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.word_embedding_matrix, start_tokens, end_token)

            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, outputs,
                                                      output_layer=tf.layers.Dense(self.vocab_size))
            outputs, states, length = tf.contrib.seq2seq.dynamic_decode(decoder)

        # 학습 모델 구축
        self.logits, self.cost, self.train_op = self.build_ops(outputs, self.targets)
        self.outputs = outputs.sample_id


    def cell(self, output_keep_prob):
        rnn_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def build_cells(self, output_keep_prob=0.5):
        # enc_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell(output_keep_prob)
        #                                         for _ in range(1)])
        # dec_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell(output_keep_prob)
        #                                         for _ in range(1)])
        # context_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell(output_keep_prob)
        #                                         for _ in range(1)])
        enc_cell = self.cell(output_keep_prob)
        dec_cell = self.cell(output_keep_prob)
        context_cell = self.cell(output_keep_prob)
        return enc_cell, dec_cell, context_cell

    def build_ops(self, outputs, targets):

        logits = outputs.rnn_output #tf.layers.dense(outputs, self.vocab_size, activation=None)

        max_time = self.get_max_time(targets)
        weights = tf.sequence_mask(self.dec_length, max_time, dtype=logits.dtype)

        cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits, targets, weights))

        # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(cost, global_step=self.global_step)

        return logits, cost, train_op

    def get_max_time(self, tensor):
        return tensor.shape[1].value

    def train(self, session, enc_input, enc_length, dec_input, dec_length, targets):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input_idx: enc_input,
                                      self.dec_input_idx: dec_input,
                                      self.enc_length: enc_length,
                                      self.dec_length: dec_length,
                                      self.targets: targets})

    def test(self, session, enc_input, enc_length, dec_input, dec_length, targets):
        prediction_check = tf.equal(self.outputs, self.targets)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))
        return session.run([self.targets, self.outputs, accuracy],
                           feed_dict={self.enc_input_idx: enc_input,
                                      self.dec_input_idx: dec_input,
                                      self.enc_length: enc_length,
                                      self.dec_length: dec_length,
                                      self.targets: targets})

    def predict(self, session, enc_input, enc_length, dec_input, dec_length):
        return session.run(self.outputs,
                           feed_dict={self.enc_input_idx: enc_input,
                                      self.dec_input_idx: dec_input,
                                      self.enc_length: enc_length,
                                      self.dec_length: dec_length
                                      })