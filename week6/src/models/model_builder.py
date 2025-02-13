import copy

import os
import torch
import torch.nn as nn
# from transformers.models.bert.modeling_bert import BertModel
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint["optim"][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != "-1":
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == "adam") and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model"
                + " but optimizer state is empty"
            )

    else:
        optim = Optimizer(
            args.optim,
            args.lr,
            args.max_grad_norm,
            beta1=args.beta1,
            beta2=args.beta2,
            decay_method="noam",
            warmup_steps=args.warmup_steps,
        )

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint["optims"][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != "-1":
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == "adam") and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model"
                + " but optimizer state is empty"
            )

    else:
        optim = Optimizer(
            args.optim,
            args.lr_bert,
            args.max_grad_norm,
            beta1=args.beta1,
            beta2=args.beta2,
            decay_method="noam",
            warmup_steps=args.warmup_steps_bert,
        )

    params = [
        (n, p) for n, p in list(model.named_parameters()) if n.startswith("bert.model")
    ]
    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint["optims"][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != "-1":
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == "adam") and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model"
                + " but optimizer state is empty"
            )

    else:
        optim = Optimizer(
            args.optim,
            args.lr_dec,
            args.max_grad_norm,
            beta1=args.beta1,
            beta2=args.beta2,
            decay_method="noam",
            warmup_steps=args.warmup_steps_dec,
        )

    params = [
        (n, p)
        for n, p in list(model.named_parameters())
        if not n.startswith("bert.model")
    ]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(nn.Linear(dec_hidden_size, vocab_size), gen_func)
    generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()

        # ----------------------------------------------------------------------------------------
        # 补全代码
        # 加载 Bert 的预训练模型，赋值给 self.model
        # 加载的模型可以参考 https://huggingface.co/models?filter=zh
        # ----------------------------------------------------------------------------------------
        # '../../models/chinese_roberta_wwm_ext_pytorch',
        # 本地模型路径 E:\代码\models\chinese-roberta-wwm-ext
        model_path = 'E:\代码\models\chinese-roberta-wwm-ext-large' \
            if large else 'E:\代码\models\chinese-roberta-wwm-ext'
        self.finetune = finetune
        self.model = BertModel.from_pretrained(model_path, cache_dir=temp_dir, return_dict=False)


    def forward(self, x, segs, mask):
        if self.finetune:
            # 输出last_hidden_state
            top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(
            self.bert.model.config.hidden_size,
            args.ext_ff_size,
            args.ext_heads,
            args.ext_dropout,
            args.ext_layers,
        )

        if args.encoder == "baseline":
            bert_config = BertConfig(
                self.bert.model.config.vocab_size,
                hidden_size=args.ext_hidden_size,
                num_hidden_layers=args.ext_layers,
                num_attention_heads=args.ext_heads,
                intermediate_size=args.ext_ff_size,
            )
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if args.max_pos > 512:
            # 修改position_embeddings
            my_pos_embeddings = nn.Embedding(
                args.max_pos, self.bert.model.config.hidden_size
            )
            my_pos_embeddings.weight.data[:512] = \
                self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = \
                self.bert.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(
                args.max_pos - 512, 1
            )
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
            self.bert.model.embeddings.register_buffer(
                "position_ids", torch.arange(args.max_pos).expand((1, -1)) # position_ids不需要被更新，因此放入buffer
            )

        if checkpoint is not None:
            self.load_state_dict(checkpoint["model"], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        # ----------------------------------------------------------------------------------------
        # 补全代码
        # 给 bert 增加一层 self.ext_layer，完成 BertSum 的前向传播过程
        # 需要注意的是，clss 是 <CLS> 所在位置，mask_cls 为真实的抽取出来句子的位置
        # 以上两个值需要仔细去看 data_loader.py 中的 Batch 类
        # ----------------------------------------------------------------------------------------
        # json文件中，将每一条数据的文本按标点符号拆成小句，如 “你好，更换全车油水，机油。变速箱油，刹车油，防冻液，清洗节气门，进气管，燃烧室，三元催化。”
        # 变为 [["你", "好"], ["更", "换", "全", "车", "油", "水"], ["机", "油"], ["变", "速", "箱", "油"], ["刹", "车", "油"], ["防", "冻", "液"], ["清", "洗", "节", "气", "门"], ["进", "气", "管"], ["燃", "烧", "室"], ["三", "元", "催", "化"]]
        # 然后在这些小句子的前后加上<CLS>和<SEP>标记，组成一个长句，而clss则是长句子中每个小句子的CLS的位置。
        # mask_cls 则是 clss 中抽取出来句子的位置
        # print(src[0])
        top_vec = self.bert(src, segs, mask_src)

        # 利用ext_layer对所有句子的cls位置的向量进行判断，选择适合作为摘要的句子

        # pytorch的索引中，[a, b]表示从a行中选出b列
        # 如果b是高维的话，那么a的第一个维度要和b的第一个维度对应，如b是[5,20]的向量，那么a就要是[5,1]
        sent_clss = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss] # [batch_size, sub_sents, d_model]
        sents_vec = sent_clss * mask_cls[:, :, None].float() # 由于clss的填充使用的index是0，会与位置id 0冲突，因此必须进行mask，将填充的位置置为0向量
        sent_scores = self.ext_layer(sents_vec, mask_cls) # [batch_size, sub_sents]


        return sent_scores, mask_cls


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict(
                    [
                        (n[11:], p)
                        for n, p in bert_from_extractive.items()
                        if n.startswith("bert.model")
                    ]
                ),
                strict=True,
            )

        if args.encoder == "baseline":
            print(f'Bert encoder training from scratch ...')
            bert_config = BertConfig(
                self.bert.model.config.vocab_size,
                hidden_size=args.enc_hidden_size,
                num_hidden_layers=args.enc_layers,
                num_attention_heads=8,
                intermediate_size=args.enc_ff_size,
                hidden_dropout_prob=args.enc_dropout,
                attention_probs_dropout_prob=args.enc_dropout,
                return_dict=False
            )
            self.bert.model = BertModel(bert_config)

        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(
                args.max_pos, self.bert.model.config.hidden_size
            )
            my_pos_embeddings.weight.data[
            :512
            ] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[
            512:
            ] = self.bert.model.embeddings.position_embeddings.weight.data[-1][
                None, :
                ].repeat(
                args.max_pos - 512, 1
            )
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(
            self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0
        )
        if self.args.share_emb:
            tgt_embeddings.weight = copy.deepcopy(
                self.bert.model.embeddings.word_embeddings.weight
            )

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size,
            heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size,
            dropout=self.args.dec_dropout,
            embeddings=tgt_embeddings,
        )

        self.generator = get_generator(
            self.vocab_size, self.args.dec_hidden_size, device
        )
        self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint["model"], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if args.use_bert_emb:
                tgt_embeddings = nn.Embedding(
                    self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0
                )
                tgt_embeddings.weight = copy.deepcopy(
                    self.bert.model.embeddings.word_embeddings.weight
                )
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
