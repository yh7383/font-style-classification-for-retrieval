import torch, gc
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import lmdb

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
        input = {'cmds': data['cmds'].to(device),
                 'args': data['args'].to(device)
                 }
        output = transformer_encoder(input)
        # 直接提取特征
        output = output.contiguous().view(opts.batch_size, -1)
        # # 提取均值特征
        # mean = torch.mean(output, dim=2)
        # std = torch.std(output, dim=2)
        # output = torch.cat((mean, std), dim=1)
    return output

def compute_Cosine(output235, base_ziti_ts):
    output235 = output235.to(device)
    base_ziti_ts = base_ziti_ts.to(device)
    return F.cosine_similarity(output235.unsqueeze(1), base_ziti_ts.unsqueeze(0), dim=2)

def topk_smallest_indices(tensor, k):
    # 使用torch.topk获取前k个最小值的索引，第二个参数负数表示取最小值
    values, indices = torch.topk(tensor, k, largest=True)
    return indices

def get_ktop_idx_type(ktop_idx, lmdb_path):
    lmdb_env = lmdb.open(lmdb_path)
    txn = lmdb_env.begin(write=True)
    ktop_type = torch.tensor([])
    for i in range(ktop_idx.size(0)):
        value = txn.get(str(ktop_idx[i].item()).encode())
        data = pickle.loads(value)
        ktop_type = torch.cat((ktop_type, torch.tensor([data['type']])))
    return ktop_type

def compute_mAPk(ktop_type, real_type):
    same_num = 0
    mAPk = 0
    for i in range(ktop_type.size(0)):
        if ktop_type[i].item() == real_type:
            same_num += 1
            mAPk += same_num/(i+1)
    if mAPk == 0:
        return mAPk
    else:
        return mAPk/same_num

def test_50ziti(opts):
    # load data
    test_load = get_loader(opts.root_path, opts.max_len, opts.d_model, opts.batch_size, opts.num_workers, 'test_build')
    base_load = get_loader(opts.root_path, opts.max_len, opts.d_model, opts.batch_size, opts.num_workers, 'base')

    # load model
    transformer_encoder = TransformerEncoder(opts.num_layers, opts.d_model, opts.nhead, opts.dim_feedforward, opts.max_len, opts.num_types, opts.dropout, device, 'test_build')
    transformer_encoder = transformer_encoder.to(device)
    transformer_encoder = load_checkpoint(transformer_encoder, filename=opts.load_model_path)

    k = 10
    all_mAPk = 0
    count = 0

    for idx1, data1 in enumerate(test_load):
        # 得到data1关于235维的评分情况(64,256*32)
        output235 = netForward(data1, transformer_encoder, opts) # 得到输入的235分布(64X235)
        # 存储所有得分(64,datalen)
        output_cos = []
        for idx2, data2 in enumerate(base_load):
            # 计算与base的Cosine（得分）
            output_cos_64 = compute_Cosine(output235, data2['model_output'])
            # 将结果添加到列表中
            output_cos.append(output_cos_64)

        # 将列表中的结果连接起来
        output_cos = torch.cat(output_cos, dim=1)

        # 遍历batch_size内的tensor，每个做得分排序
        for i in range(output235.size(0)):
            print('font type: ', data1['type'][i].item())
            print('font encoding: ', data1['font_name'])
            # print('output cosine similarity: ', output_cos[i])
            # real_type从236开始，
            ktop_idx = topk_smallest_indices(output_cos[i], k)
            print('ktop_idx: ', ktop_idx)
            ktop_type = get_ktop_idx_type(ktop_idx, './data/standard_net_ziti/50UFSC_235')
            print('ktop_type: ', ktop_type)
            mAPk = compute_mAPk(ktop_type, data1['type'][i].item())
            all_mAPk += mAPk
            count += 1
            print('mAPk:', str(mAPk))

    print('mAP@k: ', all_mAPk/count)


def main():
    opts = get_parser_main_model().parse_args()
    # 考虑创建日志文件------------------------------------------
    test_50ziti(opts)

if __name__ == '__main__':
    main()