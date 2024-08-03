from math import pi, log
from functools import wraps
from multiprocessing import context
from textwrap import indent
import math, copy
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from train_embedding.options import get_parser_main_model

opts = get_parser_main_model().parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)


class SVGEmbedding(nn.Module):
    def __init__(self):
        super(SVGEmbedding, self).__init__()
        # DE设置为512
        # psembeddingzidian: 128
        embed_dim = 256
        self.command_embed = nn.Embedding(4, 256)
        self.arg_embed = nn.Embedding(512, 256)
        self.embed_fcn = nn.Linear(embed_dim * 8, embed_dim)
        self.pos_encoding = PositionalEncoding(d_model=10, max_len=256)
        self._init_embeddings()



    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

    def forward(self, commands, args, groups=None):
        # deepvecfontv2
        # S, GN, _ = commands.shape
        # embed_cmds = self.command_embed(commands.squeeze().long())
        # embed_args = self.embed_fcn(self.arg_embed((args).long()).view(S, GN, -1))
        # src = embed_cmds + embed_args
        # # src: bs64, sq512, dm256
        # src = src.permute(1, 0, 2)
        # # src: seq_len, batch_size, d_emb
        # src = self.pos_encoding(src)

        # other
        S, GN, _ = commands.shape
        # embed_cmds = self.command_embed(commands.squeeze().long())
        embed_cmds1 = commands / 10 + 0.1
        embed_cmds2 = commands / 10 + 0.5
        embed_cmds = torch.cat((embed_cmds1, embed_cmds2), dim=2)
        embed_args = args / 256
        src = torch.cat((embed_cmds, embed_args), dim=2)
        # src = embed_cmds + embed_args
        # src: bs64, sq512, dm256
        src = src.permute(1, 0, 2)
        # src: seq_len, batch_size, d_emb
        src = self.pos_encoding(src)
        return src


class ProType(nn.Module):
    def __init__(self):
        super(ProType, self).__init__()
        self.linear = nn.Linear(10 * 256, 235)
        self.bn = nn.BatchNorm1d(235)
        self.ReLU = nn.ReLU()

    def forward(self, src):
        # src: sl256, bs64, dm512
        src = src.permute(1, 2, 0)
        # src: bs64, dm512, sl256
        src = src.contiguous().view(src.size(0), -1)
        # src: bs64, -1
        output = self.linear(src)
        output = self.bn(output)
        # output = self.ReLU(output)
        # output: bs64, type235
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, model):
        super(TransformerEncoder, self).__init__()
        self.model = model
        self.embedding = SVGEmbedding()
        self.conv = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.ReLU1 = nn.ReLU()
        self.pos_encoding = PositionalEncoding(d_model=16, max_len=256)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(10, 5, 2048, opts.dropout),
            opts.num_layers
        )
        self.protype = ProType()

    def forward(self, src, mask=None):
        cmds, args = src['cmds'].to(device), src['args'].to(device)
        src = self.embedding(cmds, args)
        # # 包含CNN
        # # src: sl512, bs64, dm256
        # src = src.permute(1, 2, 0)
        # # src: bs64, dm256, sl512
        # src = self.conv(src)
        # # src: bs64, dm512, sl256
        # src = self.bn1(src)
        # src = self.ReLU1(src)
        # # src: bs64, dm512, sl256
        # src = src.permute(2, 0, 1)
        # # src: sl256, bs64, dm512
        # # src = self.pos_encoding(src)
        # 无CNN
        # src: sl512, bs64, dm256
        output = self.transformer_encoder(src, mask)
        if self.model == 'train' or self.model == 'test':
            # output: sl256, bs64, dm512
            output = self.protype(output)
            return output
        else:
            # output: sl256, bs64, dm512
            output = output.permute(1, 0, 2)
            # output: bs64, sl256, dm512
            return output
