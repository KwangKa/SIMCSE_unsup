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

#### 训练 
```bash
python train_unsup.py ./data/news_title.txt ./path/to/huggingface_pretrained_model
```

*详细参数*
```
usage: train_unsup.py [-h] [--pretrained PRETRAINED] [--model_out MODEL_OUT]
                      [--num_proc NUM_PROC] [--max_length MAX_LENGTH]
                      [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
                      [--tao TAO] [--device DEVICE]
                      [--display_interval DISPLAY_INTERVAL]
                      [--save_interval SAVE_INTERVAL] [--pool_type POOL_TYPE]
                      [--dropout_rate DROPOUT_RATE]
                      train_file

positional arguments:
  train_file            train text file

optional arguments:
  -h, --help            show this help message and exit
  --pretrained PRETRAINED
                        huggingface pretrained model (default: hfl/chinese-
                        bert-wwm-ext)
  --model_out MODEL_OUT
                        model output path (default: ./model)
  --num_proc NUM_PROC   dataset process thread num (default: 5)
  --max_length MAX_LENGTH
                        sentence max length (default: 100)
  --batch_size BATCH_SIZE
                        batch size (default: 64)
  --epochs EPOCHS       epochs (default: 2)
  --lr LR               learning rate (default: 1e-05)
  --tao TAO             temperature (default: 0.05)
  --device DEVICE       device (default: cuda)
  --display_interval DISPLAY_INTERVAL
                        display interval (default: 50)
  --save_interval SAVE_INTERVAL
                        save interval (default: 100)
  --pool_type POOL_TYPE
                        pool_type (default: cls)
  --dropout_rate DROPOUT_RATE
                        dropout_rate (default: 0.3)
```

#### 测试
```bash
python test_unsup.py

query title:基金亏损路未尽 后市看法仍偏谨慎

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
