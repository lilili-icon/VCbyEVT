# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gensim

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        modle = gensim.models.KeyedVectors.load_word2vec_format(dataset+'/data/' + embedding,binary=True)
        self.embedding_pretrained = modle if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.4                                              # 随机失活 0
        self.require_improvement = 3000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 200                                           # epoch数
        self.batch_size = 128                                          # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 300

        # self.embed = 300 self.embedding_pretrained.size(1)\
        #     if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            vocab_size = len(config.embedding_pretrained.index_to_key) + 1
            vector_size = config.embedding_pretrained.vector_size
            # 随机生成weight
            weight = torch.randn(vocab_size, vector_size)

            words = config.embedding_pretrained.index_to_key

            word_to_idx = {word: i + 1 for i, word in enumerate(words)}
            # 定义了一个unknown的词.
            word_to_idx['<unk>'] = 0
            idx_to_word = {i + 1: word for i, word in enumerate(words)}
            idx_to_word[0] = '<unk>'

            for i in range(len(config.embedding_pretrained.index_to_key)):
                try:
                    index = word_to_idx[config.embedding_pretrained.index_to_key[i]]
                except:
                    continue
                vector = config.embedding_pretrained.get_vector(idx_to_word[word_to_idx[config.embedding_pretrained.index_to_key[i]]])
                weight[index, :] = torch.from_numpy(vector)

            self.embedding = nn.Embedding.from_pretrained(weight,freeze=True)

        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
