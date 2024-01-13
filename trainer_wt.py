import torch
import numpy as np
import os
import configargparse
import time
import pickle
from tqdm import tqdm, trange
import signal
import math
import json
import sys
from utils import *
from dataloaders import *
from torch.utils.tensorboard import SummaryWriter
from networks import create_mlp, create_net
from MetaTemplate import MetaTemplate
import maml_wt as maml 
from multi_modules import *
import matplotlib.pyplot as plt
from losses import plot_curve
import MI



torch.backends.cudnn.benchmark = True

def train_meta_template(args):
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    block_size = args.block_size
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # for tensorboard
    # writer = SummaryWriter(os.path.join(basedir, expname))

    dataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44875.vti', 'v02')
    dataset.read_volume_data()
    a = BlockGenerator(dataset.get_volume_data(), dataset.get_volume_res())
    data_block_array = a.generate_data_block_in_center(block_size, method=args.block_gen_method,
                                             block_num=args.block_num)
    
    # save blocks position xyz
    blocks = [block.v for block in data_block_array]
    path = os.path.join(basedir, expname, 'data_block_array.raw')
    np.array(blocks).astype(float).tofile(f"{path}")

    groups = maml.maml_init(args.groups_num, data_block_array, args, context=args.group_init)
    blocks_loss = [[] for i in range(len(data_block_array))]#记录每个epoch的blocks的loss
    templates_exchange = []
    templates = None
    for j in range(args.repeat_num):
        # TODO 考虑多线程生成
        templates = maml.maml_template_generate(groups, data_block_array, args, templates)
        groups_old = groups
        groups, losses = maml.maml_reassignment(args, data_block_array, templates, num_query_steps=args.query_steps,
                                   query_lrate=args.query_lrate, repeat_num=j)

        #记录loss和交换次数
        for i in range(len(losses)):
            blocks_loss[i].append(losses[i])
        sum = 0
        for i in range(len(groups_old)):
            if(i<len(groups)):
                a = [x for x in groups[i] if x in groups_old[i]]#groups[i]与groups_old[i]中相同的元素
                b = len(groups[i]) - len(a)#新增
                c = len(groups_old[i]) - len(a)#消失
                sum = sum + b + c
            else:
                sum = sum +len(groups_old[i])
        sum = sum/2
        templates_exchange.append(sum)

        # 保存groups和losses
        path = os.path.join(basedir, expname, f'epoches_{j:06d}')
        os.makedirs(path, exist_ok=True)

        filepath = os.path.join(path, f'epoches_{j:03d}_groups.txt')
        print('store groups:',groups)
        with open(filepath,'wb') as file:
            pickle.dump(groups,file)

        filepath = os.path.join(path, f'epoches_{j:03d}_losses.txt')
        print('store losses:',losses)
        with open(filepath,'wb') as file:
            pickle.dump(losses,file)
        

        if (j+1) % args.i_weights == 0 or j == 0:
            for i, [template, _] in enumerate(templates):
                os.makedirs(os.path.join(basedir, expname, 'epoches_{:06d}'.format(j+1)), exist_ok=True)
                path = os.path.join(basedir, expname, 'epoches_{:06d}'.format(j+1), '{:06d}_template.tar'.format(i))
                torch.save({
                    'network_fn_state_dict': template.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
        if groups == groups_old:
            break

    return blocks_loss, templates_exchange

def train_meta_template_multi_timestamp(args):
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    block_size = args.block_size

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # for tensorboard
    # writer = SummaryWriter(os.path.join(basedir, expname))

    # 生成data_block_array
    vDataset = MultiTimeStampDataset('asteroid', 'v02')
    vDataset.read_data()
    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    data_block_array = block_generator.generate_data_block(block_size, method=args.block_gen_method,
                                             block_num=args.block_num)

    # save blocks volume
    blocks = [block.v for block in data_block_array]
    path = os.path.join(basedir, expname, 'data_block_array.raw')
    np.array(blocks).astype(float).tofile(f"{path}")

    print("templates initial begin")
    groups = maml.maml_init(args.groups_num, data_block_array, args, context=args.group_init)
    print("templates initial complete.")

    # if args.vtk_off_screen_draw:
    vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                    os.path.join(basedir, expname, 'blocks_array.png'))

    blocks_loss = [[] for i in range(len(data_block_array))]  # 记录每个epoch的blocks的loss
    templates_exchange = []
    templates = None
    groups_old = None
    for j in range(args.repeat_num):
        # TODO 生成
        print("groups' templates generate begin")
        templates = maml.maml_template_generate(groups, data_block_array, args, templates, groups_old, j)
        print("groups' templates generate complete")

        ## calculate latent vector MI
        print('calculate templates heat map begin')
        for i in trange(args.MI_epoches):
            templates_latent_vector = maml.get_templates_vector(templates, 3, args)
            templates_heat_map = maml.calculate_templates_heat_map(templates_latent_vector)
            maml.reduce_templates_MI(templates, templates_heat_map)
        print('calculate templates heat map complete')
        
        print('test templates begin')
        maml.test_templates_fitting(groups, data_block_array, templates, j, args)
        print("test templates complete")

        print('blocks reassignment begin')
        groups_old = groups
        groups, losses = maml.maml_reassignment(args, data_block_array, templates, num_query_steps=args.query_steps,
                                   query_lrate=args.query_lrate,repeat_num=j)
        print("blocks reassignment complete")

        # 记录loss和交换次数
        for i in range(len(losses)):
            blocks_loss[i].append(losses[i])

        sum = 0
        for i in range(len(groups_old)):
            if i<len(groups) :
                a = [x for x in groups[i] if x in groups_old[i]]#groups[i]与groups_old[i]中相同的元素
                b = len(groups[i]) - len(a)#新增
                c = len(groups_old[i]) - len(a)#消失
                sum = sum + b + c
            else:
                sum = sum +len(groups_old[i])
        templates_exchange.append(sum)
        print('exchange number:', sum)

        # 保存groups和losses
        path = os.path.join(basedir, expname, f'epoches_{j:06d}')
        os.makedirs(path, exist_ok=True)

        filepath = os.path.join(path, f'groups.txt')
        print('store groups:',groups)
        with open(filepath,'wb') as file:
            pickle.dump(groups,file)

        filepath = os.path.join(path, f'losses.txt')
        print('store losses:',losses)
        with open(filepath,'wb') as file:
            pickle.dump(losses,file)

        psnr = [10*math.log(1/loss,10) for loss in losses]
        print('psnr:', psnr)

        if j % args.i_weights == 0 or j == 0 or groups_old == groups:
            for i, [template, _] in enumerate(templates):
                os.makedirs(os.path.join(basedir, expname, 'epoches_{:06d}'.format(j)), exist_ok=True)
                path = os.path.join(basedir, expname, 'epoches_{:06d}'.format(j), '{:06d}_template.tar'.format(i))
                torch.save({
                    'network_fn_state_dict': template.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
        torch.cuda.empty_cache()
        if groups == groups_old:
            break

    return blocks_loss, templates_exchange

def draw(blocks_loss, templates_exchange, args):
    print("blocks loss:", blocks_loss)
    print("templates total exchange:", templates_exchange)
    x = np.arange(len(templates_exchange))
    plt.figure()
    plt.title('blocks loss')
    plt.xlabel('repeat')
    plt.ylabel('losses')
    for i in range(len(blocks_loss)):
        y = np.array(blocks_loss[i])
        plt.plot(x,y)
    path = os.path.join(args.basedir, args.expname, "blocks_loss.png")
    plt.savefig(path)
    plt.clf()
    plt.figure()
    plt.title('groups exchange number')
    plt.xlabel('epoch')
    plt.ylabel('total number')
    y = np.array(templates_exchange)
    plt.plot(x,y)
    path = os.path.join(args.basedir, args.expname, "templates_exchange.png")
    plt.savefig(path)
    plt.clf()
    return

def test_maml_and_query(args):
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44875.vti', 'v02')
    vDataset.read_volume_data()
    data_block_size = args.block_size
    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    data_block_array = block_generator.generate_data_block_with_offset(data_block_size, args.block_num)
    if args.vtk_off_screen_draw:
        vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                        os.path.join(basedir, expname, 'blocks_array.png'))

    templates = None
    groups_old = None
    data_block_array_train = data_block_array[0:args.block_num-1]
    data_block_array_test = [data_block_array[-1]]
    groups = maml.maml_init(args.groups_num, data_block_array_train, args, context=args.group_init)
    templates = maml.maml_template_generate(groups, data_block_array_train, args, templates, groups_old)
    # 画出一个template
    path = os.path.join(basedir, expname, 'template.tar')
    torch.save({
        'network_fn_state_dict': templates[0][0].state_dict(),
    }, path)
    print('Saved checkpoints at', path)
    # if args.vtk_off_screen_draw:
    #     vtk_draw_templates(templates, data_block_size, args.vtk_off_screen_draw,
    #                        os.path.join(basedir, expname, 'templates_epoches_{:04d}.png'.format(args.maml_epoches)))
    for i, block_i in enumerate(data_block_array_test):
        data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                  shuffle=True, num_workers=2)
        os.makedirs(os.path.join(basedir, expname, 'block_{:06d}'.format(i + 1)), exist_ok=True)
        template = deepcopy(templates[0])
        loss_run, steps, loss_d = maml.maml_optimize_template_draw(template[0], data_loader, template[1],
                                                              args.query_steps, args.query_lrate,
                                                              os.path.join(basedir, expname, 'block_{:06d}/'.format(i + 1)))
        

        kong_template = create_net(args)
        loss_kong, steps_kong, loss_d_kong = maml.maml_optimize_template_draw(kong_template[0], data_loader,
                                                                         kong_template[1],
                                                                         args.query_steps, args.query_lrate)

        plot_curve(steps, loss_d, 'steps', 'psnr', x2_vals=steps_kong, y2_vals=loss_d_kong,
                   legend=['template', 'without_template'],
                   path=os.path.join(basedir, expname, 'block_{:06d}/'.format(i + 1)))
        
def test_template(args):
    basedir = args.basedir
    expname = args.expname

    vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44875.vti', 'v02')
    vDataset.read_volume_data()
    data_block_size = args.block_size
    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    data_block_array = block_generator.generate_data_block_in_center(data_block_size, 30)

    if args.vtk_off_screen_draw:
        vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                        os.path.join(basedir, expname, 'blocks_array.png'))

    # groups = maml_init(args.groups_num, data_block_array, args, context=args.group_init)
    # templates = None
    # groups_old = None
    # templates = maml_template_generate(groups, data_block_array, args, templates, groups_old)
    # 画出一个template
    path = os.path.join(basedir, expname, 'epoches_000005','000000_template.tar')

    templates = create_net(args)

    templates[0].load_state_dict(torch.load(path)['network_fn_state_dict'])
    templates = [templates]
    data_block_array = [data_block_array[-1]]
    for i, block_i in enumerate(data_block_array):
        df = open(os.path.join(basedir, expname, 'training_losses.txt'), mode='wb')
        data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                  shuffle=True, num_workers=2)
        vtk_draw_blocks([block_i])
        template = deepcopy(templates[0])
        loss_run, steps, loss_d = maml.maml_optimize_template_draw(template[0], data_loader, template[1],
                                                              args.query_steps, args.query_lrate)

        template = deepcopy(templates[0])
        loss_run, steps, loss_d_without_decay = maml.maml_optimize_template_draw(template[0], data_loader, template[1],
                                                              args.query_steps, args.query_lrate, lrate_decay=False)

        os.makedirs(os.path.join(basedir, expname, 'block_template_{:06d}'.format(i + 1)), exist_ok=True)

        kong_template = create_net(args)
        loss_kong, steps_kong, loss_d_kong = maml.maml_optimize_template_draw(kong_template[0], data_loader,
                                                                         kong_template[1],
                                                                         args.query_steps, args.query_lrate)

        kong_template = create_net(args)
        loss_kong, steps_kong, loss_d_kong_without_decay = maml.maml_optimize_template_draw(kong_template[0], data_loader,
                                                                         kong_template[1],
                                                                         args.query_steps, args.query_lrate,
                                                                         lrate_decay=False)

        pickle.dump(steps, df)
        pickle.dump(loss_d, df)
        pickle.dump(loss_d_kong, df)
        pickle.dump(loss_d_without_decay, df)
        pickle.dump(loss_d_kong_without_decay, df)
        df.close()
        plot_curve(steps, loss_d, 'Iterations', 'PSNR[dB]', x2_vals=steps_kong, y2_vals=loss_d_kong,
                   x3_vals=steps, y3_vals=loss_d_without_decay, x4_vals=steps, y4_vals=loss_d_kong_without_decay,
                   legend=['Meta-Learned Initialization', 'Standard Initialization',
                           'Meta-Learned Initialization without decay', 'Standard Initialization without decay'],
                   path=os.path.join(basedir, expname, 'block_template_{:06d}'.format(i + 1)))

                   


def main():
    args = get_args()
    if args.task == 'train_KiloNet':
        train_multi_mlp(args)
    elif args.task == 'train_KiloNet_by_templates':
        train_multi_mlp_based_templates(args)
    elif args.task == 'train_templates':
        blocks_loss, templates_exchange = train_meta_template(args)
        draw(blocks_loss, templates_exchange, args)
    elif args.task == 'train_templates_multi_timestamp':
        blocks_loss, templates_exchange = train_meta_template_multi_timestamp(args)
        draw(blocks_loss, templates_exchange, args)
    elif args.task == 'test':
        # test_maml_and_query(args)
        test_template(args)
    else:
        print("No task to do!")

if __name__ == '__main__':
    main()