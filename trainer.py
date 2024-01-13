import time

import torch
from torch.utils.tensorboard import SummaryWriter
from maml import *
from multi_modules import *
from losses import plot_curve
from renderer import VolumeRender
import time
import pickle

torch.backends.cudnn.benchmark = True


def train_mlp(args):
    # 载入数据
    dataset = AsteroidDataset('pv_insitu_300x300x300_25982.vti', 'prs', 100)
    dataset.read_volume_data()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    writer = SummaryWriter(os.path.join(basedir, expname))

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create MLP model
    start, network_query_fn, grad_vars, optimizer, model = create_mlp(args)
    global_step = start
    n_iters = 1000
    print('Begin')
    start = start + 1

    for i in trange(start, n_iters):
        time0 = time.time()
        loss_running = 0.0
        for batch_x, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_x = batch_x.to(torch.float32).to(get_device(args.GPU))
                batch_y = batch_y.to(torch.float32).to(get_device(args.GPU))

            res_fn = network_query_fn(batch_x, model)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res_fn, batch_y)
            train_loss = torch.mean(torch.abs(res_fn-batch_y))

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()
            loss_running += train_loss.item()/(len(train_loader)/args.batch_size)
        # NOTE: IMPORTANT!
        writer.add_scalar('train_loss', loss_running, global_step=global_step)
        decay_rate = 0.1
        decay_steps = args.lrate_decay*1000
        # 学习率指数级下降
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        print("new rate:", new_lrate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time() - time0
        print(f"Step: {global_step}, Loss: {loss_running}, Time: {dt}")

        # Rest is logging
        if i % args.i_weights == 0 or i == 1:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss_running}")

        # validation
        if i % args.i_validation == 0:
            test_loss = 0.0
            num = 0
            with torch.no_grad():
                pass

        global_step += 1


# baseline
def train_multi_mlp(args):
    # TODO 来改这个
    # vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_29693.vti', 'v02')
    # vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44778.vti', 'v02')
    vDataset = AsteroidDataset('./data/ya32/pv_insitu_300x300x300_100174.vti', 'v02')
    # vDataset = AsteroidDataset('./data/ya31/pv_insitu_300x300x300_44560.vti', 'v02')
    # vDataset = AsteroidDataset('./data/ya32/pv_insitu_300x300x300_053763.vti', 'v02')
    # vDataset = AsteroidDataset('./data/yB31/pv_insitu_300x300x300_44194.vti', 'v02')
    # 读取参数
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    # 分块 volume
    # 每一块选择初始化
    vDataset.read_volume_data()
    data_block_size = args.block_size
    # 生成训练数据
    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    # 默认uniform
    data_block_array = block_generator.generate_data_block(data_block_size)
    MN = MultiNetwork(args, len(data_block_array))
    blocks = BlockSum(data_block_array)
    data_loader = torch.utils.data.DataLoader(blocks, batch_size=1,
                                              shuffle=True, num_workers=2)

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    vtk_draw_single_volume(vDataset, False, '29693.png')
    return
    df = open(os.path.join(basedir, expname, 'training_losses.txt'), mode='wb')
    epoches = 1000
    losses_plot = []
    epoches_plot = []
    for i in trange(epoches):
        loss_running = 0.0
        for batch_x, batch_y in data_loader:
            batch_x = np.squeeze(batch_x, axis=0)
            batch_y = np.squeeze(batch_y, axis=0)
            if torch.cuda.is_available():
                batch_x = torch.Tensor(batch_x).to(torch.float32).to(get_device(args.GPU))
                batch_y = torch.Tensor(batch_y).to(torch.float32).to(get_device(args.GPU))
            res = MN(batch_x)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res, batch_y)
            MN.optimizer_zero_grad()
            loss.backward()
            MN.optimizer_step()

            loss_running += loss.item()/len(data_loader)
        losses_plot.append(mse2psnr(torch.tensor(loss_running)))
        epoches_plot.append(i)
        # TODO 加上学习率衰减
        decay_rate = 0.1
        decay_steps = 500
        # decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        MN.decay_lrate(new_lrate)
        tqdm.write(f"[Multi MLP] Iter: {i} Loss: {loss_running}")
        if i % args.i_weights == 0:
            create_path(os.path.join(basedir, expname,
                                     'yb31_44194_', '{:06d}'.format(i)))
            if args.vtk_off_screen_draw:
                vtk_draw_multi_modules(MN, data_block_size, [300, 300, 300], args.vtk_off_screen_draw,
                                       os.path.join(basedir, expname,
                                        'yb31_44194_','{:06d}'.format(i), 'modules.png'))
            MN.saved_checkpoints(os.path.join(basedir, expname,'yb31_44194_',
                                              '{:06d}'.format(i)))
    # plot loss
    pickle.dump(losses_plot, df)
    pickle.dump(epoches_plot, df)
    df.close()
    plot_curve(epoches_plot, losses_plot,
               'Iterations', 'PSNR(dB)',
               path=os.path.join(basedir, expname))


def train_meta_model(args):
    # 创建网络模型
    dataset_1 = AsteroidDataset('pv_insitu_300x300x300_23878.vti', 'prs')
    dataset_1.read_volume_data()
    train_loader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    dataset_2 = AsteroidDataset('pv_insitu_300x300x300_25982.vti', 'prs')
    dataset_2.read_volume_data()
    train_loader_2 = torch.utils.data.DataLoader(dataset_2, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    dataset_3 = AsteroidDataset('pv_insitu_300x300x300_26766.vti', 'prs')
    dataset_3.read_volume_data()
    train_loader_3 = torch.utils.data.DataLoader(dataset_3, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    start, network_query_fn, grad_vars, optimizer, model = create_mlp(args)
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    # maml = MetaTemplate(model, network_query_fn, MSE_loss)
    data_block_array = [
        dataset_1, dataset_2, dataset_3
    ]
    meta_dataset = MetaDataset(data_block_array)

    basedir = args.basedir
    # only for test
    expname = args.expname+'_meta_test'
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    writer = SummaryWriter(os.path.join(basedir, expname))

    num_parameters = count_parameters(model)

    print(f'\n\nTraining model with {num_parameters} parameters\n\n')

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    total_steps = 0
    epoches = 100

    train_loader_meta = torch.utils.data.DataLoader(meta_dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=args.num_workers)

    for _ in trange(start, epoches):
        # for i, set_ in enumerate(meta_dataset):
        for batch_input in train_loader_meta:
            # forward function 应该输出batch的数据量
            # x_spt, y_spt, x_qry, y_qry = set_['context'][0], set_['context'][1], set_['query'][0], set_['query'][1]
            # if torch.cuda.is_available():
            #     x_spt = torch.from_numpy(x_spt).to(torch.float32).to(get_device(args.GPU))
            #     y_spt = torch.from_numpy(y_spt).to(torch.float32).to(get_device(args.GPU))
            #     x_qry = torch.from_numpy(x_qry).to(torch.float32).to(get_device(args.GPU))
            #     y_qry = torch.from_numpy(y_qry).to(torch.float32).to(get_device(args.GPU))
            #
            # print(maml(x_spt, y_spt, x_qry, y_qry))
            print(batch_input)
            meta_split(batch_input)

        total_steps += 1


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
    writer = SummaryWriter(os.path.join(basedir, expname))

    # dataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44875.vti', 'v02')
    # dataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_29693.vti', 'v02')
    # dataset = EcmwfDataset('201810_all_variables.nc', 't', 10)
    dataset = CombustionDataset()
    dataset.read_volume_data()
    a = BlockGenerator(dataset.get_volume_data(), dataset.get_volume_res(), args.block_chunk)
    data_block_array = a.generate_data_block(block_size, method=args.block_gen_method,
                                             block_num=args.block_num)

    # 绘制block
    vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                    os.path.join(basedir, expname, 'blocks_array.png'))

    groups = maml_init(args.groups_num, data_block_array, args, context=args.group_init)
    templates = None

    for j in range(args.repeat_num):
        # TODO 生成
        templates = maml_template_generate(groups, data_block_array, args, templates)

        # 绘制templates
        vtk_draw_templates(templates, block_size, args.vtk_off_screen_draw,
                           os.path.join(basedir, expname, 'templates_epoches_{:04d}.png'.format(j)))

        temp_i = 0
        for group in groups:
            blocks_temp = [data_block_array[block_id] for block_id in group]
            if len(blocks_temp) != 0:
                vtk_draw_blocks(blocks_temp, args.vtk_off_screen_draw,
                                os.path.join(basedir, expname, 'blocks_epoches{:04d}_{:04d}.png'.format(j, temp_i)))
                temp_i += 1

        groups_old = groups
        groups = maml_reassignment(data_block_array, templates, num_query_steps=args.query_steps,
                                   query_lrate=args.query_lrate)

        if (j+1) % args.i_weights == 0 or j == 0 or groups_old == groups:
            for i, [template, _] in enumerate(templates):
                os.makedirs(os.path.join(basedir, expname, 'epoches_{:06d}'.format(j+1)), exist_ok=True)
                path = os.path.join(basedir, expname, 'epoches_{:06d}'.format(j+1), '{:06d}_template.tar'.format(i))
                torch.save({
                    'network_fn_state_dict': template.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
        if groups == groups_old:
            break


def train_multi_mlp_based_templates(args):
    # TODO 验证 template可以加速的训练时间
    basedir = args.basedir
    expname = args.expname
    epoches_name = 'epoches_000028'
    # 载入template

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, epoches_name, f) for f in sorted(os.listdir(os.path.join(basedir,
                expname, epoches_name))) if 'tar' in f]

    print("Found ckpts", ckpts)
    templates = []
    for ckpt_path in ckpts:
        model, network_query_fn = create_net(args)
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['network_fn_state_dict'])
        templates.append([model, network_query_fn])
        res = template_to_volume([model, network_query_fn], vec3f(50))
        res = np.array(res)
        print(res)

    # 训练数据
    # vDataset = AsteroidDataset('pv_insitu_300x300x300_29693.vti', 'v02')
    df = open(os.path.join(basedir, expname, epoches_name, '44194_psnr.txt'), mode='wb')
    # 用这个做测试
    # vDataset = AsteroidDataset('./data/ya11/pv_insitu_300x300x300_26886.vti', 'v02')
    # vDataset = AsteroidDataset('./data/ya31/pv_insitu_300x300x300_44560.vti', 'v02')
    # vDataset = AsteroidDataset('./data/ya32/pv_insitu_300x300x300_053763.vti', 'v02')
    # vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_29693.vti', 'v02')
    vDataset = AsteroidDataset('./data/yB31/pv_insitu_300x300x300_44194.vti', 'v02')
    vDataset.read_volume_data()
    vtk_draw_single_volume(vDataset, True, '44194.png')
    data_block_size = args.block_size

    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    # 默认uniform
    data_block_array = block_generator.generate_data_block(data_block_size)
    blocks = BlockSum(data_block_array)
    data_loader = torch.utils.data.DataLoader(blocks, batch_size=1,
                                              shuffle=True, num_workers=2)

    # 创建网络
    MN = MultiNetwork(args, len(data_block_array))
    # 按照template初始化网络
    MN.initialize_multi_mlp(templates, data_loader, query_steps=args.query_steps, query_lrate=args.query_lrate)

    epoches = 1000
    psnr_plot = []
    epoches_plot = []

    for i in trange(epoches):
        loss_running = 0.0
        num = 0.0
        for batch_x, batch_y in data_loader:
            batch_x = np.squeeze(batch_x, axis=0)
            batch_y = np.squeeze(batch_y, axis=0)
            if torch.cuda.is_available():
                batch_x = torch.Tensor(batch_x).to(torch.float32).to(get_device(args.GPU))
                batch_y = torch.Tensor(batch_y).to(torch.float32).to(get_device(args.GPU))
            res = MN(batch_x)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res, batch_y)

            MN.optimizer_zero_grad()
            loss.backward()
            MN.optimizer_step()
            loss_running += loss.item() * batch_x.shape[1]
            num = num + batch_x.shape[1]
        loss_running = loss_running / num
        psnr_plot.append(mse2psnr(torch.tensor(loss_running)))
        epoches_plot.append(i)
        tqdm.write(f"[Multi MLP] Iter: {i} Loss: {loss_running}")
        # TODO 加上学习率衰减
        decay_rate = 0.1
        decay_steps = 400
        # decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        MN.decay_lrate(new_lrate)
        if i % args.i_weights == 0:
            create_path(os.path.join(basedir, expname,
                                     epoches_name, 'yb31_44194_', '{:06d}'.format(i)))
            if args.vtk_off_screen_draw:
                vtk_draw_multi_modules(MN, data_block_size, [300, 300, 300], args.vtk_off_screen_draw,
                                       os.path.join(basedir, expname,
                                       epoches_name, 'yb31_44194_','{:06d}'.format(i), 'modules.png'))
            MN.saved_checkpoints(os.path.join(basedir, expname, epoches_name, 'yb31_44194_',
                                              '{:06d}'.format(i)))

        # 画出来
    plot_curve(epoches_plot, psnr_plot, 'Iterations', 'PSNR(dB)',
               path=os.path.join(basedir, expname, epoches_name))
    pickle.dump(psnr_plot, df)
    pickle.dump(epoches_plot, df)
    df.close()
    return MN


def train_meta_template_multi_timestamp(args):
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
    vDataset = MultiTimeStampDataset('asteroid', 'v02')
    vDataset.read_data()

    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)

    data_block_array = block_generator.generate_data_block(block_size, method=args.block_gen_method,
                                             block_num=args.block_num)

    groups = maml_init(args.groups_num, data_block_array, args, context=args.group_init)

    if args.vtk_off_screen_draw:
        vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                        os.path.join(basedir, expname, 'blocks_array.png'))

    templates = None
    groups_old = None
    for j in range(args.repeat_num):
        # TODO 生成
        templates = maml_template_generate(groups, data_block_array, args, templates, groups_old)
        torch.cuda.empty_cache()
        if args.vtk_off_screen_draw:
            # 绘制templates
            vtk_draw_templates(templates, block_size, args.vtk_off_screen_draw,
                               os.path.join(basedir, expname, 'templates_epoches_{:04d}.png'.format(j)))

            temp_i = 0
            for i, group in enumerate(groups):
                blocks_temp = [data_block_array[block_id] for block_id in group]
                res_blocks = []
                for block_i in blocks_temp:
                    template_temp = deepcopy(templates[temp_i])
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
                    vtk_draw_blocks(blocks_temp, args.vtk_off_screen_draw,
                                    os.path.join(basedir, expname, 'blocks_epoches{:04d}_{:04d}.png'.format(j, temp_i)))
                    vtk_draw_templates(res_blocks, block_size, args.vtk_off_screen_draw,
                                    os.path.join(basedir, expname,
                                                 'blocks_epoches_optimize_{:04d}_{:04d}.png'.format(j, temp_i)))
                    temp_i += 1

        groups_old = groups
        groups = maml_reassignment(data_block_array, templates, num_query_steps=args.query_steps,
                                   query_lrate=args.query_lrate)

        if (j + 1) % args.i_weights == 0 or j == 0 or groups_old == groups:
            if groups_old == groups:
                args.maml_epoches = 200
                templates = maml_template_generate(groups, data_block_array, args, templates, None)
            for i, [template, _] in enumerate(templates):
                os.makedirs(os.path.join(basedir, expname, 'epoches_{:06d}'.format(j + 1)), exist_ok=True)
                path = os.path.join(basedir, expname, 'epoches_{:06d}'.format(j + 1), '{:06d}_template.tar'.format(i))
                torch.save({
                    'network_fn_state_dict': template.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
        if groups == groups_old:
            break


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
    # data_block_array = block_generator.generate_data_block_in_center(data_block_size, 30)
    data_block_array = block_generator.generate_data_block_with_offset(data_block_size, args.block_num)
    if args.vtk_off_screen_draw:
        vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                        os.path.join(basedir, expname, 'blocks_array.png'))

    templates = None
    groups_old = None
    data_block_array_train = data_block_array[0:args.block_num-1]
    data_block_array_test = [data_block_array[-1]]
    groups = maml_init(args.groups_num, data_block_array_train, args, context=args.group_init)
    templates = maml_template_generate(groups, data_block_array_train, args, templates, groups_old)
    # 画出一个template
    path = os.path.join(basedir, expname, 'template.tar')
    torch.save({
        'network_fn_state_dict': templates[0][0].state_dict(),
    }, path)
    print('Saved checkpoints at', path)
    if args.vtk_off_screen_draw:
        vtk_draw_templates(templates, data_block_size, args.vtk_off_screen_draw,
                           os.path.join(basedir, expname, 'templates_epoches_{:04d}.png'.format(args.maml_epoches)))
    data_block_array_test = [data_block_array_test[-1]]
    for i, block_i in enumerate(data_block_array_test):
        data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                  shuffle=True, num_workers=2)
        template = deepcopy(templates[0])
        loss_run, steps, loss_d = maml_optimize_template_draw(template[0], data_loader, template[1],
                                                              args.query_steps, args.query_lrate)
        os.makedirs(os.path.join(basedir, expname, 'block_template_{:06d}'.format(i + 1)), exist_ok=True)

        kong_template = create_net(args)
        loss_kong, steps_kong, loss_d_kong = maml_optimize_template_draw(kong_template[0], data_loader,
                                                                         kong_template[1],
                                                                         args.query_steps, args.query_lrate)

        plot_curve(steps, loss_d, 'steps', 'psnr', x2_vals=steps_kong, y2_vals=loss_d_kong,
                   legend=['template', 'without_template'],
                   path=os.path.join(basedir, expname, 'block_template_{:06d}'.format(i + 1)))


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
    path = os.path.join(basedir, expname, 'template.tar')

    templates = create_net(args)

    templates[0].load_state_dict(torch.load(path)['network_fn_state_dict'])
    templates = [templates]
    data_block_array = [data_block_array[-1]]
    for i, block_i in enumerate(data_block_array):
        df = open(os.path.join(basedir, expname, 'training_losses.txt'), mode='wb')
        data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                  shuffle=True, num_workers=2)
        template = deepcopy(templates[0])
        loss_run, steps, loss_d = maml_optimize_template_draw(template[0], data_loader, template[1],
                                                              args.query_steps, args.query_lrate)

        template = deepcopy(templates[0])
        loss_run, steps, loss_d_without_decay = maml_optimize_template_draw(template[0], data_loader, template[1],
                                                              args.query_steps, args.query_lrate, lrate_decay=False)

        os.makedirs(os.path.join(basedir, expname, 'block_template_{:06d}'.format(i + 1)), exist_ok=True)

        kong_template = create_net(args)
        loss_kong, steps_kong, loss_d_kong = maml_optimize_template_draw(kong_template[0], data_loader,
                                                                         kong_template[1],
                                                                         args.query_steps, args.query_lrate)

        kong_template = create_net(args)
        loss_kong, steps_kong, loss_d_kong_without_decay = maml_optimize_template_draw(kong_template[0], data_loader,
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


def draw_templates(args):
    basedir = args.basedir
    expname = args.expname

    vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44875.vti', 'v02')
    vDataset.read_volume_data()
    data_block_size = args.block_size
    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    data_block_array = block_generator.generate_data_block_in_center(data_block_size, 30)

    kong_template = create_net(args)
    kong_template[0].load_state_dict(torch.load(os.path.join(basedir, expname, 'template.tar'))
                                     ['network_fn_state_dict'])

    if args.vtk_off_screen_draw:
        vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                        os.path.join(basedir, expname, 'blocks_array.png'))

    data_block_array_l = [data_block_array[-1]]
    for i, block_i in enumerate(data_block_array_l):
        template = deepcopy(kong_template)
        data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                  shuffle=True, num_workers=2)

        maml_optimize_template_draw(template[0], data_loader,
                                    template[1],
                                    args.query_steps, args.query_lrate,
                                    path=os.path.join(args.basedir, args.expname, 'template'))

        template_n = create_net(args)
        maml_optimize_template_draw(template_n[0], data_loader,
                                    template_n[1],
                                    args.query_steps, args.query_lrate,
                                    path=os.path.join(args.basedir, args.expname, 'sin'))
        template_relu = create_mlp_maml(args)

        maml_optimize_template_draw(template_relu[0], data_loader,
                                    template_relu[1],
                                    args.query_steps, args.query_lrate,
                                    path=os.path.join(args.basedir, args.expname, 'relu'))
    # vtk_draw_templates(block_draw, data_block_size, True,
    #                    os.path.join(args.basedir, args.expname, 'block_opti_{:06d}.png'.format(args.query_steps)))
    # vtk_draw_templates(block_without, data_block_size,
    #                    True,
    #                    os.path.join(basedir, args.expname, 'block_notemplate_opti_{:06d}.png'.format(args.query_steps)))


def draw_templates_(args):
    model = create_net(args)
    # vtk_draw_templates([model], [150,150,150], True, '1.png')
    vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44875.vti', 'v02')
    vDataset.read_volume_data()
    data_block_size = args.block_size
    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    data_block_array = block_generator.generate_data_block_in_center(data_block_size, 30)
    data_loader = torch.utils.data.DataLoader(data_block_array[0], batch_size=1,
                                              shuffle=True, num_workers=2)
    time_start = time.time()
    maml_optimize_template(model[0], data_loader, model[1], 200, 1e-4)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')


def main():
    # 设置随机种子
    np.random.seed(0)
    args = get_args()
    if args.task == 'train_KiloNet':
        train_multi_mlp(args)
    elif args.task == 'train_KiloNet_by_templates':
        train_multi_mlp_based_templates(args)
    elif args.task == 'train_templates':
        train_meta_template(args)
    elif args.task == 'train_templates_multi_timestamp':
        train_meta_template_multi_timestamp(args)
    elif args.task == 'test':
        test_maml_and_query(args)
        # test_template(args)
    else:
        print("No task to do!")
        draw_templates(args)


if __name__ == '__main__':
    main()