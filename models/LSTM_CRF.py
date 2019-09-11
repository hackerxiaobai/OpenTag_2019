import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from  .basic_module import BasicModule
from .squeeze_embedding import SqueezeEmbedding


class NERLSTM_CRF(BasicModule):
    def __init__(self, opt, embedding_dim=200, hidden_dim=300, dropout=0.2, word2id=100000, tag2id=4):
        super(NERLSTM_CRF, self).__init__()

        self.opt = opt
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = word2id + 1
        self.tag_to_ix = tag2id
        self.tagset_size = tag2id

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        #CRF
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=False)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def forward(self, inputs):
        x, att, tags = inputs
        #CRF
        x = x.transpose(0,1)

        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        #CRF
        outputs = self.crf.decode(outputs)
        return outputs

    def log_likelihood(self, inputs):
        x, att, tags = inputs
        x = x.transpose(0,1)
        tags = tags.transpose(0,1)
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return - self.crf(outputs, tags)
