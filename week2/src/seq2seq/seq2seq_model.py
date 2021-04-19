# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.seq2seq.model_layers import Encoder, BahdanauAttention, LuongAttention, Decoder
from src.utils.gpu_utils import config_gpu
from src.utils.params_utils import get_params
from src.utils.wv_loader import load_embedding_matrix, Vocab


class Seq2Seq(nn.Module):
    def __init__(self, params, vocab):
        super(Seq2Seq, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.vocab = vocab
        self.batch_size = params["batch_size"]
        self.enc_units = params["enc_units"]
        self.dec_units = params["dec_units"]
        self.attn_units = params["attn_units"]
        self.rnn_type = params['rnn_type']
        self.encoder = Encoder(embedding_matrix=self.embedding_matrix,
                               enc_units=self.enc_units,
                               rnn_type=self.rnn_type)

        self.att_model = 'bahdanau'
        self.decoder = Decoder(embedding_matrix=self.embedding_matrix,
                               dec_units=self.dec_units,
                               rnn_type='gru',
                               att_model=self.att_model,
                               hidden_size=self.dec_units,
                               units=self.dec_units
                               )

    def teacher_decoder(self, dec_hidden, enc_output, dec_target):
        '''
        :param dec_hidden:
        :param enc_output: (batch_size, sen_len, enc_hidden)
        :param dec_target: (batch_size, sen_len)
        :return:           (batch_size, sen_len-1, vocab)
        '''
        predictions = []

        dec_input = torch.unsqueeze([self.vocab.START_DECODING_INDEX] * self.batch_size, 1) # (batch_size, 1)

        #  Teacher forcing- feeding the target as the next input
        for t in range(1, dec_target.shape[1]):
            # passing enc_output to the decoder
            """
            应用decoder来一步一步预测生成词语概论分布
            请写出你的代码
            如：xxx = self.decoder(), 采用Teachering Forcing方法
            """
            dec_input = torch.unsqueeze(dec_target[:, t], 1)
            pred, state, attention_weights = self.decoder(dec_input, dec_hidden, enc_output)
            # pred shape == (batch_size, vocab)
            # prediction list == (sen_len-1) * (batch_size, vocab)
            predictions.append(pred)

        # return shape == (batch_size, sen_len-1, vocab)
        return torch.stack(predictions, 1), dec_hidden


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 获得参数
    params = get_params()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # 计算vocab size
    input_sequence_len = 200

    params = {"vocab_size": vocab.count,
              "embed_size": 500,
              "enc_units": 512,
              "attn_units": 512,
              "dec_units": 512,
              "batch_size": 128,
              "input_sequence_len": input_sequence_len}

    model = Seq2Seq(params, vocab)

    # example_input
    example_input_batch = tf.ones(shape=(params['batch_size'], params['input_sequence_len']), dtype=tf.int32)

    # sample input
    sample_hidden = model.encoder.initialize_hidden_state()

    sample_output, sample_hidden = model.encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    sample_decoder_output, _, _ = model.decoder(tf.random.uniform((params['batch_size'], 1)),
                                                sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
