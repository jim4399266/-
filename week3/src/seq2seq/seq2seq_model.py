# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.seq2seq.model_layers import Encoder, Decoder
from src.seq2seq.attentions import BahdanauAttention, LuongAttention
# from src.utils.gpu_utils import config_gpu
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

        self.encoder = Encoder(embedding_matrix=self.embedding_matrix,
                               batch_size=self.batch_size,
                               enc_units=self.enc_units,
                               rnn_type='gru')

        # self.att_model = 'bahdanau'
        self.att_model = BahdanauAttention(att_hidden_size=self.dec_units,
                               att_units=self.dec_units)

        self.decoder = Decoder(embedding_matrix=self.embedding_matrix,
                               dec_units=self.dec_units,
                               rnn_type='gru',
                               att_model=self.att_model,
                               )

    def teacher_decoder(self, dec_hidden, enc_output, dec_target):
        '''
        :param dec_hidden: (batch_size, hidden_size)
        :param enc_output: (batch_size, sen_len, enc_hidden)
        :param dec_target: (batch_size, sen_len)
        :return:           (batch_size, sen_len-1, vocab)
        '''
        predictions = []

        dec_input = torch.unsqueeze(
            torch.tensor([self.vocab.START_DECODING_INDEX] * self.batch_size), 1).cuda() # (batch_size, 1)
        # dec_input = torch.unsqueeze([self.vocab.START_DECODING_INDEX] * self.batch_size, 1)

        #  Teacher forcing- feeding the target as the next input
        for t in range(1, dec_target.shape[1]):
            # passing enc_output to the decoder
            """
            应用decoder来一步一步预测生成词语概论分布
            请写出你的代码
            如：xxx = self.decoder(), 采用Teachering Forcing方法
            """
            pred, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_output)
            dec_input = torch.unsqueeze(dec_target[:, t], 1)
            # pred shape == (batch_size, vocab)
            # prediction list == (sen_len-1) * (batch_size, vocab)
            predictions.append(pred)

        # return shape == (batch_size, sen_len-1, vocab)
        return torch.stack(predictions, 1), dec_hidden


if __name__ == '__main__':
    # GPU资源配置, pytorch 不用配置
    # config_gpu()
    # 获得参数
    params = get_params()
    # 设置训练设备
    device = torch.device('cuda:0'
                          if torch.cuda.is_available() and params['device']=='cuda' else 'cpu')
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # 计算vocab size
    input_sequence_len = 200

    params = {"vocab_size": vocab.count,
              # "embed_size": 500,
              "enc_units": 512,
              "attn_units": 512,
              "dec_units": 512,
              "batch_size": 128,
              "input_sequence_len": input_sequence_len}

    model = Seq2Seq(params, vocab).to(device)

    # example_input
    example_input_batch = torch.ones(size=(params['batch_size'], params['input_sequence_len']), dtype=torch.int32, device=device)

    sample_output, sample_hidden = model.encoder(example_input_batch)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(params['dec_units'], params['attn_units']).to(device)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    sample_decoder_output, _, _ = model.decoder(torch.ones(size=(params['batch_size'], 1),
                                                                         dtype=torch.int32, device=device),
                                                sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
