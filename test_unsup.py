# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka

import logging
from SimCSERetrieval import SimCSERetrieval


def main():
    fname = "./data/news_title.txt"
    pretrained = "hfl/chinese-bert-wwm-ext"  # huggingface modelhub 下载的预训练模型
    simcse_model = "./model/batch-1400"
    batch_size = 64
    max_length = 100
    device = "cuda"

    logging.info("Load model")
    simcse = SimCSERetrieval(fname, pretrained, simcse_model, batch_size, max_length, device)

    logging.info("Sentences to vectors")
    simcse.encode_file()

    logging.info("Build faiss index")
    simcse.build_index(n_list=1024)
    simcse.index.nprob = 20

    query_sentence = "基金亏损路未尽 后市看法仍偏谨慎"
    print("\nquery title:{0}".format(query_sentence))
    print("\nsimilar titles:")
    print(u"\n".join(simcse.sim_query(query_sentence, topK=10)))


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

