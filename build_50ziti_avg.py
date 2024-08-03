import torch
import torch.optim
import torch.nn.functional as F
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

def build_50zitiidx(opts):
    base_load = get_loader(opts.root_path, opts.max_len, opts.d_model, opts.batch_size, opts.num_workers, 'build')

    transformer_encoder = TransformerEncoder(opts.num_layers, opts.d_model, opts.nhead, opts.dim_feedforward, opts.max_len, opts.num_types, opts.dropout, device)
    transformer_encoder = transformer_encoder.to(device)
    transformer_encoder = load_checkpoint(transformer_encoder, filename=opts.load_model_path)

    base_ziti = {}
    base_count = {}
    # 计算UFSC字体的235type
    for idx, data in enumerate(base_load):
        output235 = netForward(data, transformer_encoder, opts)
        label50 = data['type']
        for i in range(label50.size(0)):
            label = label50[i].item()
            if label in base_ziti:
                base_ziti[label] += output235[i]
                base_count[label] += 1
            else:
                base_ziti[label] = output235[i]
                base_count[label] = 1
        if idx % opts.print_batch_every == 0:
            print(f'{idx} complish transform 235 type')
    # 计算UFSC字体的235type的平均值
    for i in range(len(base_ziti)):
        base_ziti[i+236] = base_ziti[i+236]/base_count[i+236]

    # 指定保存的文件名
    file_path = './data/standard_net_ziti/50UFSC.pkl'

    # 使用 pickle 库将数据保存到 .pkl 文件中
    with open(file_path, 'wb') as f:
        pickle.dump(base_ziti, f)

def main():
    opts = get_parser_main_model().parse_args()
    # 考虑创建日志文件------------------------------------------
    build_50zitiidx(opts)

if __name__ == '__main__':
    main()