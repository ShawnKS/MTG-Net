"""
SelfAttentionEncoder.py
"""

import torch, torch.optim, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
# from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from torch.autograd import Variable, set_detect_anomaly
from torch.utils.data import (
    DataLoader, Dataset, RandomSampler, SubsetRandomSampler, Subset, SequentialSampler
)
from torch.utils.data.distributed import DistributedSampler

from math import floor
import numpy as np
import copy
from copy import deepcopy
import sys
from Layer import TransformerLayer

# A naive Linear Implementation
class HOINet(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=128, out_dim=8):
        super().__init__()
        self.fc_layer1 = nn.Linear(in_dim, hidden_dim)
        self.inter_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(
        self,x
    ):
        hidden_1 = self.relu(self.fc_layer1(x))
        hidden_2 = self.relu((self.inter_layer3(self.relu((self.inter_layer2(self.relu((self.inter_layer1(hidden_1)))))))))
        output = self.fc_layer2(hidden_2)
        return output


class HOINetTransformer(nn.Module):
    def __init__(self,
               num_layers=2,
               task_dim = 8,
               model_dim=128,
               num_heads=2,
               ffn_dim=128,
               dropout=0.0):
        super(HOINetTransformer, self).__init__()
        self.model_dim = model_dim
        self.first_layer = TransformerLayer(model_dim = model_dim, num_heads = num_heads, ffn_dim=ffn_dim, dim_per_head = model_dim, dropout=dropout,task_dim = task_dim)
        self.encoder_layers = nn.ModuleList(
          [TransformerLayer(model_dim = model_dim, num_heads = num_heads, dim_per_head = model_dim, ffn_dim=ffn_dim, dropout=dropout, task_dim = task_dim).to('cuda:0') for _ in
           range(num_layers)] )
        self.task_output = [nn.Linear(ffn_dim, 1).to('cuda:1') for _ in range(task_dim)]
        self.final_output = nn.Linear(ffn_dim, 1)
        # task_embedding_parameter =torch.Tensor(8,model_dim)
        # torch.nn.init.uniform(task_embedding_parameter, a=-1, b=1)
        self.task_embedding = nn.Embedding(task_dim,model_dim)
        # self.task_embedding = torch.nn.Parameter(task_embedding_parameter)
        # self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        # self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        # pos_embedding -> meta _feture

    def forward(self, inputs,index):
        # output = self.seq_embedding(inputs)
        output = self.task_embedding(index)
        # output.detach()
        test_output = self.task_embedding(index)
        # output += self.pos_embedding(inputs_len)

        # self_attention_mask = padding_mask(inputs, inputs)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output,inputs)
            selected_mask = torch.ones(len(inputs[0])).to('cuda:1')
            # selected_mask[[1,3,6]] = 1
            test_output, test_attention = encoder(test_output[0].unsqueeze(0), selected_mask  )
            attentions.append(attention)
        outputs = []
        task_embedding = self.task_embedding(index)[0]
        encoder_output = test_output
        encoder_output = output

        output=self.final_output(output)
        result = torch.squeeze(output,2)
        return result, attentions, task_embedding, encoder_output

class HOINetMeta(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=96, out_dim=8):
        super().__init__()
        self.fc_layer1 = nn.Linear(in_dim, hidden_dim)
        self.inter_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(
        self,x
    ):
        hidden_1 = self.relu(self.fc_layer1(x))
        hidden_2 = self.relu((self.inter_layer3(self.relu((self.inter_layer2(self.relu((self.inter_layer1(hidden_1)))))))))
        output = self.relu(self.fc_layer2(hidden_2))
        return output

class HOINetCNN(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=96, out_dim=8):
        super().__init__()
        self.fc_layer1 = nn.Linear(in_dim, hidden_dim)
        self.inter_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.inter_layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(
        self,x
    ):
        hidden_1 = self.relu(self.fc_layer1(x))
        hidden_2 = self.relu((self.inter_layer3(self.relu((self.inter_layer2(self.relu((self.inter_layer1(hidden_1)))))))))
        output = self.relu(self.fc_layer2(hidden_2))
        return output

