import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from utils import *
from config_parser import get_args
from dataloaders import *
from networks import *
from MetaTemplate import MetaTemplate
from copy import deepcopy
import multiprocessing as mp
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,ALL_COMPLETED
from losses import mse2psnr
from losses import plot_curve
import MI

def maml_init(group_num, blocks, args, context='rand_init'):
    groups = [[] for i in range(group_num)]
    if context is 'rand_init':
        group_index = np.random.randint(low=0, high=group_num, size=len(blocks))
        for i, block in enumerate(blocks):
            groups[group_index[i]].append(i)
        return groups
    elif context is 'train_init':
        # 随机选取k个block，train之后得到对应template，在reassignment(步骤3)
        templates = []
        rand_num = generate_shuffle_number(len(blocks))
        rand_num = rand_num[:group_num]
        for num in rand_num:
            # 随机选取的block
            block_i = blocks[num]
            data = []
            data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                      shuffle=True, num_workers=2)
            for batch_x, batch_y in data_loader:
                data.append([batch_x, batch_y])
            template, fn = create_mlp_maml(args)
            maml_optimize_template(template, data, fn, optimize_steps=args.maml_epoches)
            templates.append([template, fn])
        groups,losses = maml_reassignment(args, blocks, templates, num_query_steps=args.query_steps, query_lrate=args.query_lrate)
        # groups,losses = maml_reassignment(args, blocks, templates, num_query_steps=1, query_lrate=args.query_lrate)
        return groups
    elif context is 'cluster_init':
        # TODO 根据blocks的距离来分
        pass
    else:
        raise NotImplementedError


# TODO 此函数可以用多线程来处理，同时多个group进行maml操作
def maml(group, blocks, args, template, epoches=1):
    # block 是三维数组
    block_g = []
    for index in group:
        block_g.append(blocks[index])
    # 构造类
    dataset = MetaDataset(block_g, args.maml_chunk)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                              shuffle=True, num_workers=2)
    # 创建网络的部分
    if args.model == 'mlp_relu':
        if template is None:
            model, network_query_fn = create_mlp_maml(args)
        else:
            model, network_query_fn = template
    elif args.model == 'siren':
        if template is None:
            model, network_query_fn = create_siren_maml(args)
        else:
            model, network_query_fn = template
    elif args.model == 'film_siren':
        if template is None:
            model, network_query_fn = create_film_siren_maml(args)
        else:
            model, network_query_fn = template
    else:
        raise NotImplementedError
    # TODO move loss function to create_mlp_maml
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    TemplateGenerator = MetaTemplate(model, network_query_fn, MSE_loss, meta_lr=args.meta_lr, update_lr=args.update_lr,
                                     num_meta_steps=args.meta_steps)
    # 优化一下， 先把数据输入存起来
    # data = []
    # for batch_input in data_loader:
    #     batch_input = np.squeeze(batch_input, axis=0)
    #     data.append(batch_input)
    loss = []
    steps = []
    for i in trange(0, epoches):
        loss_running = 0.0
        steps.append(i)
        for data_i in data_loader:
            # 将点 shuffle 开
            # data_i = np.squeeze(data_i, axis=0)
            # print(data_i.shape)
            data_i = data_i[:, torch.randperm(data_i.size(1))]
            set_ = meta_split(data_i, split_mode=args.meta_split_method)
            x_spt, y_spt, x_qry, y_qry = set_['context'][0], set_['context'][1], set_['query'][0], set_['query'][1]
            if torch.cuda.is_available():
                x_spt = torch.from_numpy(x_spt).to(torch.float32).to(get_device(args.GPU))
                y_spt = torch.from_numpy(y_spt).to(torch.float32).to(get_device(args.GPU))
                x_qry = torch.from_numpy(x_qry).to(torch.float32).to(get_device(args.GPU))
                y_qry = torch.from_numpy(y_qry).to(torch.float32).to(get_device(args.GPU))
            loss_running += TemplateGenerator(x_spt, y_spt, x_qry, y_qry)
            torch.cuda.empty_cache()
        loss_running /= len(data_loader)
        tqdm.write(f"[MAML] Iter: {i} Loss: {loss_running}")
        loss.append(loss_running)
        if loss_running < 1e-4:
            break
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.plot(steps,loss)
    # plt.savefig(os.path.join(args.basedir, args.expname, 'MAML_loss_{:06d}'.format(args.maml_epoches)))
    # plt.clf()
    return model, network_query_fn, steps, loss

def maml_template_generate(groups, blocks, args, templates, groups_old, repeat=-1):
    maml_templates = []
    # TODO 转化为多进程,加快运行效率
    for i, group in tqdm(enumerate(groups), total=len(groups)):
        if group:
            if groups_old is None or groups_old[i] != group:
                print("group size:",len(group))
                if args.MI_R is 'I':
                    maml_template, fn, steps, loss = maml(group, blocks, args, templates[i] if templates is not None else None,
                                             epoches=args.maml_epoches)
                else:
                    maml_template, fn, steps, loss = maml(group, blocks, args, None,
                                             epoches=args.maml_epoches)
                maml_templates.append([maml_template, fn])
                # draw maml loss curve
                os.makedirs(os.path.join(args.basedir, args.expname, f'epoches_{repeat:06d}'), exist_ok=True)
                plot_curve(x_vals=steps, y_vals=loss, x_label='Iteration', y_label='Loss', 
                           path=os.path.join(args.basedir, args.expname, f'epoches_{repeat:06d}'),
                           legend=[f'group size:{len(group)}'],
                           filename=f'template_{i:02d}_MAML_loss.png')
            else:
                maml_templates.append(templates[i])
    return maml_templates

def reduce_templates_MI(templates, templates_heat_map, lrate=1e-5):
    for i, template in enumerate(templates):
        grad_vars = list(template[0].parameters())
        optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9,0.999))
        loss = torch.tensor(np.sum(templates_heat_map[i]), requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return

def maml_optimize_template(template, data, fn, optimize_steps, lrate=1e-3):
    grad_vars = list(template.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9,0.999), weight_decay=0.0001)
    loss_running = 0.0
    losses=[]
    for i in range(optimize_steps):
        loss_running = 0.0
        for batch_x, batch_y in data:
            batch_x = np.squeeze(batch_x, axis=0)
            batch_y = np.squeeze(batch_y, axis=0)
            if torch.cuda.is_available():
                batch_x = torch.Tensor(batch_x).to(torch.float32).to(get_model_device(template))
                batch_y = torch.Tensor(batch_y).to(torch.float32).to(get_model_device(template))
            res_fn = fn(batch_x, template)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res_fn, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_running += loss.item()/len(data)
        losses.append(loss_running)
        # TODO 加上学习率衰减
    return loss_running, losses


# num_query_steps 设置为 maml inner loop steps 最好
def maml_reassignment(args, blocks, templates, num_query_steps, query_lrate, repeat_num=-1):
    groups = [[] for i in range(len(templates))]
    losses = [0 for i in range(len(blocks))]
    randint = random.randint(0,len(blocks))
    # 将所有template整合成AdaptiveMultiMLP
    # TODO 每个block用多线程加速
    for i, block_i in tqdm(enumerate(blocks), total=len(blocks)):
        loss_i = []
        if i == randint:
            losses_i = []

        # TODO MINER的方法加速
        # 拿出训练数据
        data = []
        data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                  shuffle=True, num_workers=2)
        for batch_x, batch_y in data_loader:
            data.append([batch_x, batch_y])

        for j, [template_j, fn] in enumerate(templates):
            # important! deepcopy template_j
            template_j_temp = deepcopy(template_j)
            loss_i_j, losses_i_j = maml_optimize_template(template_j_temp, data, fn, num_query_steps, lrate=query_lrate)
            loss_i.append(loss_i_j)
            if i == randint:
                losses_i.append(losses_i_j)
            del template_j_temp
        min_j = np.argmin(np.array(loss_i))
        losses[i] = np.min(np.array(loss_i))

        # 画单个block每个template的损失曲线
        if i == randint:
            plt.figure()
            plt.title(f'block{i} loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            x = np.arange(len(losses_i[0]))
            for j in range(len(losses_i)):
                y = np.array(losses_i[j])
                plt.plot(x,y,label=f'{j}')
            plt.legend()
            os.makedirs(os.path.join(args.basedir, args.expname, 'loss_img'), exist_ok=True)
            plt.savefig(os.path.join(args.basedir,args.expname,f'loss_img/repeat{repeat_num}block{i}.jpg'))
            plt.clf()

        tqdm.write(f"[reassignment] block: "
                   f"{i} min_Loss: {np.min(np.array(loss_i))}, max_loss: {np.max(np.array(loss_i))}")

        groups[min_j].append(i)

    return groups, losses

def maml_optimize_template_draw(template, data, fn, optimize_steps, lrate=1e-2, path=None, lrate_decay=True):
    grad_vars = list(template.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9,0.999))
    loss_running = 0.0
    loss_d = []
    steps = []
    if path is not None:
        vtk_draw_templates([[template, fn]], np.array([150, 150, 150]), True,
                           path + '_{:06d}.png'.format(0))
    for t in trange(optimize_steps):
        loss_running = 0.0
        for batch_x, batch_y in data:
            batch_x = np.squeeze(batch_x, axis=0)
            batch_y = np.squeeze(batch_y, axis=0)
            if torch.cuda.is_available():
                batch_x = torch.Tensor(batch_x).to(torch.float32).to(get_model_device(template))
                batch_y = torch.Tensor(batch_y).to(torch.float32).to(get_model_device(template))
            res_fn = fn(batch_x, template)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res_fn, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_running += loss.item()/len(data)
        loss_d.append(mse2psnr(torch.tensor(loss_running)))
        steps.append(t)
        # TODO 加上学习率衰减
        if lrate_decay is True:
            decay_rate = 0.1
            decay_steps = 500
            # decay_steps = args.lrate_decay
            new_lrate = lrate * (decay_rate ** (t / decay_steps))
            # print("new rate:", new_lrate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        if path is not None and (t==0 or t==4 or t==9 or t==19 or t==49 or t==99 or t==499 or t==999):
            print('times:', t, ' ', mse2psnr(torch.tensor(loss_running)))
            vtk_draw_templates([[template, fn]], np.array([150, 150, 150]), True,
                               path+'_{:06d}.png'.format(t+1))
        vtk_draw_templates([[template, fn]], np.array([150, 150, 150]))
    return loss_running, steps, loss_d

def get_templates_vector(templates, layer_index, args):
    hidden_layer_extractor = MI.Network_Latent_Vector(templates, layer_index, args)
    hidden_layer_extractor.register_hook()
    templates_latent_vector = hidden_layer_extractor.get_latent_vectors()
    hidden_layer_extractor.remove_hook()
    return templates_latent_vector

def calculate_templates_heat_map(templates_latent_vector):
    templates_latent_vector = [vector.cpu().detach().numpy() for vector in templates_latent_vector]
    MI_calculator = MI.Variable_MI()
    MI_calculator.variables_CKA_heat_map(np.array(templates_latent_vector))
    return templates_latent_vector

def test_templates_fitting(groups, data_block_array, templates, repeat_num, args):
    print("groups size:", len(groups), "templates size:", len(templates))
    for i, template in tqdm(enumerate(templates), total=len(templates)):
        blocks_temp = [data_block_array[block_id] for block_id in groups[i]]
        res_blocks = []
        for block_i in tqdm(blocks_temp):
            template_temp = deepcopy(template)
            # TODO MINER的方法加速
            data = []
            data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                        shuffle=True, num_workers=2)
            for batch_x, batch_y in data_loader:
                data.append([batch_x, batch_y])

            maml_optimize_template(template_temp[0], data, template_temp[1], args.query_steps, args.query_lrate)
            # 得到优化后的blocks
            res_blocks.append(template_temp)
        if len(blocks_temp) != 0:
            os.makedirs(os.path.join(args.basedir, args.expname, f'repeat_{repeat_num:06d}'), exist_ok=True)
            vtk_draw_blocks(blocks_temp, args.vtk_off_screen_draw,
                            os.path.join(args.basedir, args.expname, f'repeat_{repeat_num:06d}',
                                            'origin_template{:04d}.png'.format(i)))
            vtk_draw_templates(res_blocks, args.block_size, args.vtk_off_screen_draw,
                            os.path.join(args.basedir, args.expname, f'repeat_{repeat_num:06d}',
                                            'fit_template{:04d}.png'.format(i, repeat_num)))  
            

