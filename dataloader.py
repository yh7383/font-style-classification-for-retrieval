import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import lmdb

from options import get_parser_main_model
opts = get_parser_main_model().parse_args()


def standard_cor(num):
    num += 0.4
    num /= 1.7
    num *= opts.seq_arg_dim
    if num < 0:
        print('error', num)
        num = 0
    if num >= opts.seq_arg_dim:
        print('error', num)
        num = opts.seq_arg_dim - 1
    return int(num)


class SVGDataset(data.Dataset):
    def __init__(self, root_path, max_seq_len=512, seq_feature_dim=10, transform=None, mode='train'):
        super(SVGDataset, self).__init__()
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.feature_dim = seq_feature_dim
        self.transform = transform
        if mode == 'train' or mode == 'val':
            self.lmdb_path = os.path.join(root_path, 'SFSCTraindata256')
        elif mode == 'test':
            self.lmdb_path = os.path.join(root_path, 'SFUCTraindata256')
        elif mode == 'build':
            self.lmdb_path = os.path.join(root_path, 'UFSCTraindata256')
        elif mode == 'test_build':
            self.lmdb_path = os.path.join(root_path, 'UFUCTraindata256')
        elif mode == 'base':
            self.lmdb_path = os.path.join(root_path+'standard_net_ziti/', '50UFSC_235')
        self.lmdb_env = lmdb.open(self.lmdb_path)
        self.txn = self.lmdb_env.begin(write=True)
        self.lenth = self.lmdb_env.stat()['entries']

    def __getitem__(self, index):
        # lmdbd的索引从1开始
        if self.mode == 'base':
            if index < 0 or index >= self.lenth:
                print('error')
            value = self.txn.get(str(index).encode())
            data = pickle.loads(value)
            return data
        else:
            # 采用绝对坐标
            # with self.lmdb_env.begin(write=True) as txn:
            value = self.txn.get(str(index).encode())
            temp_font = pickle.loads(value)
            data = {}
            cmds = torch.empty(0, 1)
            args = torch.empty(0, 8)
            cur_corx, cur_cory = 0, 0
            for i in range(0, self.max_seq_len * self.feature_dim, 10):
                arg = []
                if temp_font['sequence'][i:i + 4] == [1, 0, 0, 0]:
                    cmd = torch.full((1, 1), 0)
                    cur_corx = temp_font['sequence'][i + 8]
                    cur_cory = temp_font['sequence'][i + 9]
                    arg.extend([standard_cor(cur_corx), standard_cor(cur_cory)])
                    arg.extend([0, 0, 0, 0])
                    arg.extend([standard_cor(cur_corx), standard_cor(cur_cory)])
                elif temp_font['sequence'][i:i + 4] == [0, 1, 0, 0]:
                    cmd = torch.full((1, 1), 1)
                    arg.extend([standard_cor(cur_corx), standard_cor(cur_cory)])
                    arg.extend([0, 0, 0, 0])
                    cur_corx = temp_font['sequence'][i + 8]
                    cur_cory = temp_font['sequence'][i + 9]
                    arg.extend([standard_cor(cur_corx), standard_cor(cur_cory)])
                elif temp_font['sequence'][i:i + 4] == [0, 0, 1, 0]:
                    cmd = torch.full((1, 1), 2)
                    arg.extend([standard_cor(cur_corx), standard_cor(cur_cory)])
                    arg.append(standard_cor(temp_font['sequence'][i + 4]))
                    arg.append(standard_cor(temp_font['sequence'][i + 5]))
                    arg.append(standard_cor(temp_font['sequence'][i + 6]))
                    arg.append(standard_cor(temp_font['sequence'][i + 7]))
                    cur_corx = temp_font['sequence'][i + 8]
                    cur_cory = temp_font['sequence'][i + 9]
                    arg.extend([standard_cor(cur_corx), standard_cor(cur_cory)])
                else:
                    cmd = torch.full((1, 1), 3)
                    arg.extend([0, 0, 0, 0, 0, 0, 0, 0])

                # if temp_font['sequence'][i:i + 4] == [1, 0, 0, 0]:
                #     cmd = torch.full((1, 1), 0)
                #     cur_corx = temp_font['sequence'][i + 8]
                #     cur_cory = temp_font['sequence'][i + 9]
                #     arg.extend([cur_corx, cur_cory])
                #     arg.extend([0, 0, 0, 0])
                #     arg.extend([cur_corx, cur_cory])
                # elif temp_font['sequence'][i:i + 4] == [0, 1, 0, 0]:
                #     cmd = torch.full((1, 1), 1)
                #     arg.extend([cur_corx, cur_cory])
                #     arg.extend([0, 0, 0, 0])
                #     cur_corx = temp_font['sequence'][i + 8]
                #     cur_cory = temp_font['sequence'][i + 9]
                #     arg.extend([cur_corx, cur_cory])
                # elif temp_font['sequence'][i:i + 4] == [0, 0, 1, 0]:
                #     cmd = torch.full((1, 1), 2)
                #     arg.extend([cur_corx, cur_cory])
                #     arg.append(temp_font['sequence'][i + 4])
                #     arg.append(temp_font['sequence'][i + 5])
                #     arg.append(temp_font['sequence'][i + 6])
                #     arg.append(temp_font['sequence'][i + 7])
                #     cur_corx = temp_font['sequence'][i + 8]
                #     cur_cory = temp_font['sequence'][i + 9]
                #     arg.extend([cur_corx, cur_cory])
                # else:
                #     cmd = torch.full((1, 1), 3)
                #     arg.extend([0, 0, 0, 0, 0, 0, 0, 0])

                arg = torch.tensor(arg).unsqueeze(0)
                cmds = torch.cat((cmds, cmd), 0)
                args = torch.cat((args, arg), 0)

            data['cmds'] = cmds
            data['args'] = args
            data['type'] = temp_font['font_type']
            data['font_name'] = temp_font['font_name']
            return data

    def __len__(self):
        return self.lenth


def get_loader(root_path, max_seq_len, seq_feature_dim, batch_size, num_workers, mode='train'):
    # SetRange = T.Lambda(lambda X: 2 * X - 1.)  # convert [0, 1] -> [-1, 1]
    # SetRange = T.Lambda(lambda X: 1. - X )  # convert [0, 1] -> [0, 1]
    transform = T.Compose([
        T.ToTensor
    ])
    dataset = SVGDataset(root_path, max_seq_len, seq_feature_dim, transform, mode)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=num_workers, drop_last=True)
    return dataloader

if __name__ == '__main__':
    lmdb_env = lmdb.open('./data/SFUCTraindata')
    txn = lmdb_env.begin(write=True)
    value = txn.get(str(27639).encode())
    temp_font = pickle.loads(value)
    print(temp_font)