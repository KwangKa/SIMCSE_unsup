# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka


class CSECollator(object):
    def __init__(self,
                 tokenizer,
                 features=("input_ids", "attention_mask", "token_type_ids"),
                 max_len=100):
        self.tokenizer = tokenizer
        self.features = features
        self.max_len = max_len

    def collate(self, batch):
        new_batch = []
        for example in batch:
            for i in range(2):
                # 每个句子重复两次
                new_batch.append({fea: example[fea] for fea in self.features})
        new_batch = self.tokenizer.pad(
            new_batch,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return new_batch
