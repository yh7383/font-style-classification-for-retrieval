import torch
import torch.nn as nn
import math

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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, seq_len, output_dim, dropout, device, model):
        super(TransformerEncoder, self).__init__()

        self.model = model
        self.device = device
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.command_embed = nn.Embedding(4, 512)
        self.arg_embed = nn.Embedding(512, 512)
        self.embed_fcn = nn.Linear(512 * 8, 512)

        # 卷积层提高注意笔画局部特点
        self.conv = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.ReLU1 = nn.ReLU()
        self.position_encoding = PositionalEncoding(512, max_len=256)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 8, 2024, dropout),
            num_layers
        )

        self.linear = nn.Linear(512 * 256, 235)
        self.bn2 = nn.BatchNorm1d(235)


    def forward(self, src, mask=None):
        cmds, args = src['cmds'].to(device), src['args'].to(device)
        cmds = self.command_embed(cmds.squeeze(2).long())
        args = self.arg_embed(args.long()).view(64, 256, -1)
        args = self.embed_fcn(args)
        src = cmds + args
        # src = torch.cat(((cmds + 1) * 0.1, args), dim = 2)

        # # 先进性positional encoding
        # # src: bs64, sl512, dm256
        # src = src.permute(1, 0, 2)
        # # src: sl512, bs64, dm24=56
        # src = self.position_encoding(src)
        # # src: sl512, bs64, dm256
        # src = src.permute(1, 2, 0)
        # # src: bs64, dm256, sl512
        # src = self.conv(src)
        # # src: bs64, dm512, sl256
        # src = self.bn1(src)
        # src = self.ReLU1(src)
        # # src: bs64, dm512, sl256
        # src = src.permute(2, 0, 1)

        # # 后进行positional encoding
        # # src: bs64, sl512, dm256
        # src = src.permute(0, 2, 1)
        # # src: bs64, dm256, sl512
        # src = self.conv(src)
        # # src: bs64, dm512, sl256
        # src = self.bn1(src)
        # src = self.ReLU1(src)
        # # src: bs64, dm512, sl256
        # src = src.permute(2, 0, 1)
        # # src: sl512, bs64, dm24=56
        # src = self.position_encoding(src)

        # src: bs64, sl512, dm256
        src = src.permute(1, 0, 2)
        src = self.position_encoding(src)

        # src: sl256, bs64, dm512
        output = self.transformer_encoder(src, mask)
        if self.model == 'train' or self.model == 'test':
            # sl,bs,dm -> bs,sl,dm
            output = output.permute(1, 0, 2)
            output = output.contiguous().view(output.size(0), -1)
            output = self.linear(output)
            # output = self.bn2(output)
            return output
        else:
            output = output.permute(1, 0, 2)
            # output = output.contiguous().view(output.size(0), -1)
            return output