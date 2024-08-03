import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

def getACC(output, label, opts):
    batch_size = opts.batch_size
    _, predicted = torch.max(output, 1)  # 沿着第1维（out_label维度）找到最大值的索引，即预测的类别

    # 计算准确率
    correct = (predicted == label).sum().item()
    accuracy = correct / batch_size

    return accuracy


def netForward(data, transformer_encoder, opts):
    # input:[batch_size][seq_len][seq_dim]
    # enc_input:[seq_len][batch_size][seq_dim]
    label = data['type']
    output = transformer_encoder(data)

    # get loss
    output = output.to(device)
    label = label.to(device)
    cross_entropy_loss = F.cross_entropy(output, label)

    # get acc
    acc = getACC(output, label, opts)

    res = {}
    res['loss'] = cross_entropy_loss
    res['acc'] = acc
    return res

def test(opts):
    # 加载数据
    test_load = get_loader(opts.root_path, opts.max_len, opts.d_model, opts.batch_size, opts.num_workers, 'test')

    # 加载模型
    transformer_encoder = TransformerEncoder(opts.num_layers, opts.d_model, opts.nhead, opts.dim_feedforward, opts.max_len, opts.num_types, opts.dropout, device, 'test')
    transformer_encoder = transformer_encoder.to(device)
    transformer_encoder = load_checkpoint(transformer_encoder, filename=opts.load_model_path)

    # 简单验证ACC就好，不做混淆矩阵之类的
    epoch_loss = 0
    epoch_accuracy = 0
    batch_count = 0

    temp_loss = 0
    temp_accuracy = 0
    temp_count = 0

    # tensorboard记录
    writer = SummaryWriter(opts.test_tensorboard_path)
    for idx, data in enumerate(test_load):
        res = netForward(data, transformer_encoder, opts)
        loss = res['loss'].item()
        acc = res['acc']
        epoch_loss += loss
        epoch_accuracy += acc
        batch_count += 1

        temp_loss += loss
        temp_accuracy += acc
        temp_count += 1
        if idx % opts.test_batch_every == 0:
            batch_data = {
                "batch_idx": idx,
                "train_loss": temp_loss/temp_count,
                "train_acc": temp_accuracy/temp_count
            }
            writer.add_scalar('Train/loss', temp_loss/temp_count, int(idx/opts.test_batch_every))
            writer.add_scalar('Train/acc', temp_accuracy/temp_count, int(idx/opts.test_batch_every))
            temp_loss, temp_accuracy, temp_count = 0, 0, 0
            print(batch_data)

    epoch_loss /= batch_count
    epoch_accuracy /= batch_count
    test_data = {
        "train_loss": epoch_loss,
        "train_acc": epoch_accuracy
    }
    print(test_data)

def main():
    opts = get_parser_main_model().parse_args()
    # 考虑创建日志文件------------------------------------------
    test(opts)

if __name__ == '__main__':
    main()