![](https://img.shields.io/badge/license-MIT-blue)
![](https://img.shields.io/badge/Python-3.8.5-blue)
![](https://img.shields.io/badge/torch-1.4.0-green)
![](https://img.shields.io/badge/transformers-4.5.1-green)
![](https://img.shields.io/badge/datasets-1.7.0-green)
![](https://img.shields.io/badge/faiss--cpu-1.7.0-green)
![](https://img.shields.io/badge/tqdm-4.49.0-green)

<h3 align="center">
<p>A PyTorch implementation of unsupervised SimCSE </p>
</h3>

[SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)

---

### 1. 用法

#### 无监督训练 
```bash
python train_unsup.py ./data/news_title.txt ./path/to/huggingface_pretrained_model
```

*详细参数*
```bash
python train_unsup.py -h
```

#### 相似文本检索测试
```bash
python test_unsup.py
```

```
query title:
基金亏损路未尽 后市看法仍偏谨慎

sim title:
基金亏损路未尽 后市看法仍偏谨慎
海通证券：私募对后市看法偏谨慎
连塑基本面不容乐观 后市仍有下行空间
基金谨慎看待后市行情
稳健投资者继续保持观望 市场走势还未明朗
下半年基金投资谨慎乐观
华安基金许之彦：下半年谨慎乐观
楼市主导 期指后市不容乐观
基金公司谨慎看多明年市
前期乐观预期被否 基金重归谨慎
```

#### STS-B数据集训练和测试
中文STS-B数据集，详情见[这里](https://github.com/pluto-junzeng/CNSD)

```bash
# 训练
python train_unsup.py ./data/STS-B/cnsd-sts-train_unsup.txt

# 验证
python eval_unsup.py
```

|模型| STS-B dev | STS-B test|
| --- | --- | --- |
| hfl/chinese-bert-wwm-ext | 0.3326 | 0.3209 |
| simcse | 0.7499 | 0.6909 |

> 与苏剑林的[实验结果](https://spaces.ac.cn/archives/8348)接近，BERT-P1是0.3465，SIMCSE是0.6904

### 2. 参考
- [SimCSE](https://github.com/princeton-nlp/SimCSE)
- [SimCSE-Chinese](https://github.com/zhengyanzhao1997/NLP-model/tree/main/model/model/Torch_model/SimCSE-Chinese)
