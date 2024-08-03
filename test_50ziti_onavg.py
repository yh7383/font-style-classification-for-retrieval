import torch, gc
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pickle
from models.TransformerEncoder import TransformerEncoder
from options import get_parser_main_model
from dataloader import get_loader

import json
import datetime
from aim import Run, Image, Distribution
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def netForward(data, transformer_encoder, opts):
    # input:[batch_size][seq_len][seq_dim]
    # enc_input:[seq_len][batch_size][seq_dim]
    with torch.no_grad():
        input = data['sequence'].permute(1, 0, 2)
        input = input.to(device)
        output = transformer_encoder(input)
        output = F.softmax(output, dim=1)
    return output

def compute_Cosine(output235, base_ziti_ts):
    output_cos = F.cosine_similarity(output235.unsqueeze(1), base_ziti_ts.unsqueeze(0), dim=2)
    return output_cos

def topk_smallest_indices(tensor, k):
    # 使用torch.topk获取前k个最小值的索引，第二个参数负数表示取最小值
    _, indices = torch.topk(tensor, k, largest=False)
    return indices

def compute_pritype(output_cos, k):
    output_pritype = torch.tensor([])
    for i in range(output_cos.size(0)):
        temp_tensor = topk_smallest_indices(output_cos[i], k)
        temp_tensor = temp_tensor.to(device)
        output_pritype = output_pritype.to(device)
        output_pritype = torch.cat((output_pritype, temp_tensor.unsqueeze(0)), dim=0)
    return output_pritype

def pritype_acc(data, output_pritype):
    acc = 0
    count = 0
    for i in range(output_pritype.size(0)):
        real_type = data['type'][i].item() - 236
        if real_type in output_pritype[i]:
           acc += 1
        count += 1
    return acc/count

def test_50ziti(opts):
    # 指定要读取的 .pkl 文件路径
    file_path = './data/standard_net_ziti/50UFSC.pkl'
    with open(file_path, 'rb') as f:
        base_ziti = pickle.load(f)

    # dict -> tensor
    base_ziti_ts = torch.stack([tensor for tensor in base_ziti.values()])

    # load data
    test_load = get_loader(opts.root_path, opts.max_len, opts.d_model, opts.batch_size, opts.num_workers, 'test_build')

    # load model
    transformer_encoder = TransformerEncoder(opts.num_layers, opts.d_model, opts.nhead, opts.dim_feedforward, opts.max_len, opts.num_types, opts.dropout, device)
    transformer_encoder = transformer_encoder.to(device)
    transformer_encoder = load_checkpoint(transformer_encoder, filename=opts.load_model_path)

    k = 3
    all_acc = 0
    count = 0
    for idx, data in enumerate(test_load):
        output235 = netForward(data, transformer_encoder, opts) # 得到输入的235分布
        output_cos = compute_Cosine(output235, base_ziti_ts) # output_cos(64 X 50)
        output_pritype = compute_pritype(output_cos, k)
        all_acc += pritype_acc(data, output_pritype)
        count += 1
    print(all_acc/count)


def main():
    opts = get_parser_main_model().parse_args()
    # 考虑创建日志文件------------------------------------------
    test_50ziti(opts)

if __name__ == '__main__':
    main()