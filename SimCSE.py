# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka

import torch.nn as nn
from transformers import BertConfig, BertModel


class SimCSE(nn.Module):
    def __init__(self, pretrained="hfl/chinese-bert-wwm-ext", pool_type="cls", dropout_prob=0.3):
        super().__init__()
        conf = BertConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        assert pool_type in ["cls", "pooler"], "invalid pool_type: %s" % pool_type
        self.pool_type = pool_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        if self.pool_type == "cls":
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == "pooler":
            output = output.pooler_output
        return output
