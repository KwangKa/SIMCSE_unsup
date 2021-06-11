# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka


import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from SimCSE import SimCSE


class SimCSERetrieval(object):
    def __init__(self,
                 fname,
                 pretrained_path,
                 simcse_model_path,
                 batch_size=32,
                 max_length=100,
                 device="cuda"):
        self.fname = fname
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        model = SimCSE(pretrained=pretrained_path).to(device)
        model.load_state_dict(torch.load(simcse_model_path))
        self.model = model
        self.model.eval()
        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode_batch(self, texts):
        text_encs = self.tokenizer(texts,
                                   padding=True,
                                   max_length=self.max_length,
                                   truncation=True,
                                   return_tensors="pt")
        input_ids = text_encs["input_ids"].to(self.device)
        attention_mask = text_encs["attention_mask"].to(self.device)
        token_type_ids = text_encs["token_type_ids"].to(self.device)
        with torch.no_grad():
            output = self.model.forward(input_ids, attention_mask, token_type_ids)
        return output

    def encode_file(self):
        all_texts = []
        all_ids = []
        all_vecs = []
        with open(self.fname, "r", encoding="utf8") as h:
            texts = []
            idxs = []
            for idx, line in tqdm(enumerate(h)):
                if not line.strip():
                    continue
                texts.append(line.strip())
                idxs.append(idx)
                if len(texts) >= self.batch_size:
                    vecs = self.encode_batch(texts)
                    vecs = vecs / vecs.norm(dim=1, keepdim=True)
                    all_texts.extend(texts)
                    all_ids.extend(idxs)
                    all_vecs.append(vecs.cpu())
                    texts = []
                    idxs = []
        all_vecs = torch.cat(all_vecs, 0).numpy()
        id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.id2text = id2text
        self.vecs = all_vecs
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, nlist=256):
        dim = self.vecs.shape[1]
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(nlist, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def sim_query(self, sentence, topK=20):
        vec = self.encode_batch([sentence])
        vec = vec / vec.norm(dim=1, keepdim=True)
        vec = vec.cpu().numpy()
        _, sim_idx = self.index.search(vec, topK)
        sim_sentences = []
        for i in range(sim_idx.shape[1]):
            idx = sim_idx[0, i]
            sim_sentences.append(self.id2text[idx])
        return sim_sentences
