import argparse

def get_parser_main_model():
    parser = argparse.ArgumentParser()
    experiment = 'experiment_235ziti_abscor2'

    # tensorboard
    parser.add_argument('--tensorboard_path', type=str, default='./logs/tensorboard/' + experiment, help='path to tensorboard logs')

    # JSON
    parser.add_argument('--json_path', type=str, default='./logs/json/' + experiment + '.json', help='path to json file')

    # pth_tar
    parser.add_argument('--checkpoint_path', type=str, default='./logs/checkpoint/' + experiment + '/', help='path to checkpoint file')

    # data set
    parser.add_argument('--seq_arg_dim', type=int, default=512, help='load model')

    # train_pattern
    parser.add_argument('--load_model', type=bool, default=0, help='load model')
    parser.add_argument('--expected_accuracy', type=float, default=0.95, help='expected accuracy')
    parser.add_argument('--load_model_path', type=str, default='./logs/checkpoint/' + experiment + '/checkpoint_epoch_5.pth.tar', help='load model path')

    # train_loader
    parser.add_argument('--root_path', type=str, default='data/', help='data root')
    parser.add_argument('--max_len', type=int, default=256, help='max sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='enpoch batch size')
    parser.add_argument('--d_model', type=int, default=10, help='sequence feature dimension')
    parser.add_argument('--num_workers', type=int, default=64, help='num workers')

    # transformer_encoder
    parser.add_argument('--model_name', type=str, default='TransformerEncoder', help='训练字体风格分类')
    parser.add_argument('--num_layers', type=int, default=6, help='TransformerEncoder layer number')
    parser.add_argument('--nhead', type=int, default=8, help='TransformerEncoder mutiply heads')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='TransformerEncoder fc dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='TransformerEncoder dropout')

    # optimier
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.998), help='adam: beta_1, beta_2')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    # epoch train
    parser.add_argument('--epochs', type=int, default=100, help='enpoch number')
    parser.add_argument('--num_types', type=int, default=235, help='font type number')
    parser.add_argument('--print_batch_every', type=int, default=100, help='every batch num print')

    # batch test
    parser.add_argument('--test_batch_every', type=int, default=10, help='every batch num test')
    parser.add_argument('--test_tensorboard_path', type=str, default='./logs/tensorboard/test_235ziti_psencoding', help='test_tensorboard_path')
    return parser