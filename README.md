# OpenTag_2019
[Scaling Up Open Tagging from Tens to Thousands: Comprehension Empowered Attribute Value Extraction from Product Title](https://www.aclweb.org/anthology/P19-1514)
> 该论文是在[OpenTag: Open Attribute Value Extraction from Product Profiles](https://arxiv.org/pdf/1806.01264.pdf)的基础上做的改进，大体思路其实没什么特别大的亮点。模型结构如下：

![模型结构](/img/1.png)

### requirements
> 1. pytorch
> 2. pytorch-transformer
> 3. sklearn
> 4. seqeval
> 5. tqdm

### data
1. data目录下的raw.txt 是全量数据，中文品牌_适用季节.pkl 是从中抽出用来实验的小数据集
2. utils下的data_process.py 提供两种获得实验数据的方式，bert分词和不用bert分词,运行 
python data_process.py 可以得到 中文品牌_适用季节.pkl
3. dataset.py 封装了Dataset和DataLoader

### model
1. 提供了两个模型，LSTM_CRF.py 做一个baseline
2. OpenTag_2019.py 复现的是该论文的模型结构

### run
1. python main.py train --batch_size=128 即可运行