# -*- coding: utf-8 -*-
# @Time    : 2021/6/18
# @Author  : kaka

import logging
import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats
from transformers import BertTokenizer
from SimCSE import SimCSE


def load_test_data(fname, tokenizer, max_length):
    lines = open(fname, "r", encoding="utf8").read().splitlines()
    sent_a = []
    sent_b = []
    score = []
    for line in lines:
        _, sa, sb, s = line.strip().split(u"||")
        sent_a.append(sa)
        sent_b.append(sb)
        score.append(float(s))
    sent_a_encs = tokenizer(sent_a, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    sent_b_encs = tokenizer(sent_b, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return {"sent_a_encs": sent_a_encs, "sent_b_encs": sent_b_encs, "score": np.array(score)}


def eval(data, model, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        a_embed = model(
            input_ids=data["sent_a_encs"]["input_ids"].to(device),
            attention_mask=data["sent_a_encs"]["attention_mask"].to(device),
            token_type_ids=data["sent_a_encs"]["token_type_ids"].to(device),
        )
        b_embed = model(
            input_ids=data["sent_b_encs"]["input_ids"].to(device),
            attention_mask=data["sent_b_encs"]["attention_mask"].to(device),
            token_type_ids=data["sent_b_encs"]["token_type_ids"].to(device),
        )
        sim_score = F.cosine_similarity(a_embed, b_embed).cpu().numpy()
        corr = scipy.stats.spearmanr(sim_score, data["score"]).correlation
    return corr


def main():
    pretrained_model_path = "hfl/chinese-bert-wwm-ext"  # huggingface 提供的预训练模型，也可指定本地模型文件
    simcse_model_path = ""  # simcse训练得到的模型文件
    f_test = "./data/STS-B/cnsd-sts-test.txt"
    f_dev = "./data/STS-B/cnsd-sts-dev.txt"

    logging.info("Load tokenizer")
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    max_length = 100
    device = torch.device("cuda")
    test_data = load_test_data(f_test, tokenizer, max_length)
    logging.info("test data:{0}".format(len(test_data["sent_a_encs"]["input_ids"])))
    dev_data = load_test_data(f_dev, tokenizer, max_length)
    logging.info("dev data:{0}".format(len(dev_data["sent_a_encs"]["input_ids"])))

    logging.info("eval bert model")
    model = SimCSE(pretrained_model_path, "cls")
    bert_test_score = eval(test_data, model, device)
    bert_dev_score = eval(dev_data, model, device)

    logging.info("eval simcse model\n")
    model.load_state_dict(torch.load(simcse_model_path))
    simcse_test_score = eval(test_data, model, device)
    simcse_dev_score = eval(dev_data, model, device)

    logging.info(u"bert model test score:{:.4f}".format(bert_test_score))
    logging.info(u"bert model dev score:{:.4f}".format(bert_dev_score))
    logging.info(u"simcse model test score:{:.4f}".format(simcse_test_score))
    logging.info(u"simcse model dev score:{:.4f}".format(simcse_dev_score))


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
