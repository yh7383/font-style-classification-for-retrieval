import torch
import torch.optim
import torch.nn.functional as F
import pickle
import lmdb


from models.TransformerEncoder import TransformerEncoder
from options import get_parser_main_model
from dataloader import get_loader

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
    # load data
    base_load = get_loader(opts.root_path, opts.max_len, opts.d_model, opts.batch_size, opts.num_workers, 'build')

    # load model
    transformer_encoder = TransformerEncoder(opts.num_layers, opts.d_model, opts.nhead, opts.dim_feedforward, opts.max_len, opts.num_types, opts.dropout, device)
    transformer_encoder = transformer_encoder.to(device)
    transformer_encoder = load_checkpoint(transformer_encoder, filename=opts.load_model_path)

    # state env
    env = lmdb.open('./data/standard_net_ziti/50UFSC_235', map_size=100000000000)
    lmdb_idx = 0
    # 计算UFSC字体的235type
    for idx, data in enumerate(base_load):
        output235 = netForward(data, transformer_encoder, opts)
        label50 = data['type']
        font_name50 = data['font_name']

        for i in range(label50.size(0)):
            label = label50[i].item()
            font_name = font_name50[i]
            txn = env.begin(write=True)
            # 插入的数据
            put_data = {}
            put_data['type'] = label
            put_data['font_name'] = font_name
            put_data['model_output'] = output235[i]
            txn.put(str(f'{label}_{font_name}').encode(), str(lmdb_idx).encode())
            txn.put(str(lmdb_idx).encode(), pickle.dumps(put_data))
            txn.commit()
            lmdb_idx += 1
        if idx % opts.print_batch_every == 0:
            print(f'{idx} complish transform 235 type')

def main():
    opts = get_parser_main_model().parse_args()
    # 考虑创建日志文件------------------------------------------
    build_50zitiidx(opts)

if __name__ == '__main__':
    main()