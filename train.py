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

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, scheduler, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch

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

def write_logs(loss, acc, batch_count, epoch, writer):
    # 计算loss和acc
    loss /= batch_count
    acc /= batch_count
    print('complish the epoch: ' + str(epoch) + ' processing')
    print('train loss: ' + str(loss))
    print('train acc: ' + str(acc))

    writer.add_scalar('Train/loss', loss, epoch)
    writer.add_scalar('Train/acc', acc, epoch)

def write_parameters(transformer_encoder, optimizer, scheduler, epoch, writer, opts):
    # 记录在tensorboard
    for name, param in transformer_encoder.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    # 记录可读取的参数数据
    state = {
        'epoch': epoch,
        'state_dict': transformer_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    save_checkpoint(state, filename=opts.checkpoint_path + f'checkpoint_epoch_{epoch}.pth.tar')


def train(opts):

    # 加载数据
    train_load = get_loader(opts.root_path, opts.max_len, opts.d_model, opts.batch_size, opts.num_workers, 'train')

    # 初始化神经网络
    transformer_encoder = TransformerEncoder(opts.num_layers, opts.d_model, opts.nhead, opts.dim_feedforward, opts.max_len, opts.num_types, opts.dropout, device, 'train')
    transformer_encoder = transformer_encoder.to(device)

    # 初始化优化器
    all_parameters = list(transformer_encoder.parameters())
    optimizer = torch.optim.Adam(all_parameters, lr=opts.lr, betas=opts.betas)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)

    # 是否加载模型
    start_epoch = -1
    if opts.load_model:
        transformer_encoder, optimizer, scheduler, start_epoch = load_checkpoint(transformer_encoder, optimizer, scheduler, filename=opts.load_model_path)

    # 训练日志
    writer = SummaryWriter(opts.tensorboard_path)

    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置循环训练模型
    for epoch in range(start_epoch + 1, opts.epochs):
        # 初始化epoch logs
        epoch_loss = 0
        epoch_accuracy = 0
        batch_count = 0
        print('is processing epoch ' + str(epoch))
        for idx, data in enumerate(train_load):
            # run net
            res = netForward(data, transformer_encoder, opts)
            loss = res['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss and acc
            epoch_loss += loss.item()
            acc = res['acc']
            epoch_accuracy += acc
            batch_count += 1

            # 输出日志
            if idx % opts.print_batch_every == 0:
                batch_data = {
                    "batch_idx": idx,
                    "train_loss": loss.item(),
                    "train_acc": acc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                print(batch_data)

        # 更新lr
        scheduler.step(epoch_loss)

        # 保存loss和acc
        write_logs(epoch_loss, epoch_accuracy, batch_count, epoch, writer)

        # 存储参数信息
        write_parameters(transformer_encoder, optimizer, scheduler, epoch, writer, opts)

        # 达到预期正确率
        if epoch_accuracy / batch_count >= opts.expected_accuracy:
            break

    writer.close()



def main():
    opts = get_parser_main_model().parse_args()
    # 考虑创建日志文件------------------------------------------
    train(opts)

if __name__ == '__main__':
    main()