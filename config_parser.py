import configargparse
import argparse


# runtime arguments
def config_parser(parser):
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--expname', type=str, default='for_debug', help='experiment n ame')
    parser.add_argument('--basedir', type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument('--datadir', type=str, default='./data/',
                         help='input data directory')
    parser.add_argument('--netdepth', type=int, default=2,
                         help='layers in network')
    parser.add_argument('--netwidth', type=int, default=64,
                         help='channels per layer')
    parser.add_argument('--lrate', type=float, default=1e-3,
                         help='learning rate')
    parser.add_argument('--lrate_decay', type=int, default=500,
                         help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument('--netchunk', type=int, default=1024*1024,
                         help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument('--ft_path', type=str, default=None,
                         help='specific weights npy file to reload for coarse network')
    parser.add_argument('--multires', type=int, default=5,
                         help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument('--i_embed', type=int, default=0,
                         help='set 0 for default positional encoding(3D location)')
    parser.add_argument('--i_validation', type=int, default=100,
                         help='frequency of validation')
    parser.add_argument('--i_print', type=int, default=10,
                         help='frequency of console printout and metric login')
    parser.add_argument('--batch_size', type=int, default=4,
                         help="batch size of pos")
    parser.add_argument('--GPU', type=str, default='0,1,2',
                         help='which gpu is used')
    parser.add_argument('--input_ch', type=int, default=3,
                         help='the channel of input')
    parser.add_argument('--output_ch', type=int, default=1,
                         help='the channel of output')
    parser.add_argument('--input_transformer_depth', type=int, default=3,
                         help='the depth on input transformation')
    parser.add_argument('--is_train', type=int, default=1,
                         help='train or test')
    # TODO 超参数
    parser.add_argument('--repeat_num', type=int, default=101,
                         help='repeat times for maml and reassignment')
    parser.add_argument('--MI_R', type=str, default='R',
                         help='MAML算法是否需要继承原来的template')
    parser.add_argument('--group_init', type=str, default='train_init',
                         help='初始化group的方式:train_init, rand_init, cluster_init')
    parser.add_argument('--groups_num', type=int, default=8,
                         help='初始化group的数量')
    parser.add_argument('--block_size', type=int, default=[50,50,50], nargs='+',
                         help='控制block的采样分辨率')
    parser.add_argument('--maml_epoches', type=int, default=10,
                         help='maml算法的外循环次数')
    parser.add_argument('--maml_boundary', type=int, default=10,
                         help="maml更改网络训练部分")
    parser.add_argument('--MI_epoches', type=int, default=5,
                         help="控制降低mutual information循环次数")
    parser.add_argument('--meta_lr', type=float, default=1e-4,
                         help='meta learning lrate')
    parser.add_argument('--update_lr', type=float, default=1e-2,
                         help='meta lea2ning update fast weights lrate')
    parser.add_argument('--meta_steps', type=int, default=5,
                         help='maml算法内循环次数')
    parser.add_argument('--query_steps', type=int, default=3,
                         help='query steps for reassignment')
    parser.add_argument('--query_lrate', type=float, default=1e-3,
                         help='lrate for reassignment')
    parser.add_argument('--block_gen_method', type=str, default='random',
                         help='generate data block method')
    parser.add_argument('--block_num', type=int, default=256,
                         help='block num')
    parser.add_argument('--meta_split_method', type=str, default='all',
                         help='meta split method for query and support set')
    parser.add_argument('--i_weights', type=int, default=1,
                         help='frequency of template ckpt saving')
    parser.add_argument('--task', type=str, default='test',
                         help='which task for trainer.py '
                              'train_KiloNet'
                              'train_KiloNet_by_templates'
                              'train_templates'
                              'train_templates_multi_timestamp')
    parser.add_argument('--model', type=str, default='mlp_relu',
                         help='choose siren, film_siren or mlp_relu')
    parser.add_argument('--shift', action='store_true',
                         help='if use shift in the input')
    parser.add_argument('--w0', help="w0 parameter from SIREN.",
                         type=float, default=30.0)
    parser.add_argument("--use_latent", help="Whether to use latent vector.",
                         type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimension of the latent vector mapped to modulations. "
                             "If set to -1, will not use latent vector.")
    parser.add_argument('--maml_chunk', type=int, default=1024 * 16,
                         help='number of maml through network in parallel, decrease if running out of memory')
    parser.add_argument('--block_chunk', type=int, default=1024*1024,
                         help='number of block through network in parallel, decrease if running out of memory')
    parser.add_argument('--vtk_off_screen_draw', action='store_true',
                         help='if draw volume by vtk')
    parser.add_argument('--dataset', type=str, default="Argon_Bubble",
                         help='dataset name')
    parser.add_argument('--inner_part', type=str, default="F",
                         help='模型内循环更新部分')

    return 


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    config_parser(parser)
    args = parser.parse_args()
    return args

