import networks
import config_parser
import torch
import os
import pickle
import random
from utils import *
from dataloaders import *
import maml_wt as maml
from copy import deepcopy
# from maml_wt import *
# from multi_modules import *

BLOCK_SIZE  =150

def train_multi_mlp_based_templates():
    args = config_parser.get_args()
    # TODO 验证 template可以加速的训练时间
    basedir = args.basedir
    expname = args.expname
    # 载入template

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print("Found ckpts", ckpts)
    templates = []
    for ckpt_path in ckpts:
        model, network_query_fn = networks.create_mlp_maml(args)
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['network_fn_state_dict'])
        templates.append([model, network_query_fn])


def save_templates():
    print("save templates")
    #加载神经网络
    args = config_parser.get_args()
    model, network_query_fn = networks.create_siren_maml(args)
    dir= os.path.join(args.basedir, args.expname)
    epoch=input("choose template epoch:")
    epoch = int(epoch)
    num=input("choose template number range:")

    # 载入data_block_array
    path = os.path.join(args.basedir, args.expname, 'data_block_array.raw')
    data_block_array=np.fromfile(f"{path}", dtype='float')
    print(data_block_array.shape)
    data_block_array = data_block_array.reshape(int(data_block_array.size/BLOCK_SIZE**3),BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)
    print(data_block_array.shape)

    # 获取groups的id
    path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}',f'epoches_{epoch:03}_groups.txt')
    print(path)
    with open(path,'rb') as file:
        groups = pickle.load(file)
    print(groups)

    losses = [0 for i in range(0,data_block_array.shape[0])]
    #载入参数并绘制raw同时获取template与对应blocks的loss
    for i in range(0,int(num)):
        model.load_state_dict(torch.load(f"{dir}/epoches_000{epoch:03}/000{i:03}_template.tar")['network_fn_state_dict'])
        #输入坐标信息[50,50,50]
        data_block_size = vec3f(BLOCK_SIZE)
        res = template_to_volume([model,network_query_fn], data_block_size)
        # pos = get_query_coords(vec3f(-1), vec3f(1), data_block_size)
        # print("坐标:",pos)
        # pos = torch.Tensor(pos).to(torch.float32).to(get_device(args.GPU))
        # res = network_query_fn(pos, model)
        # print("结果:",res)
        # res = res.cpu().detach().numpy()
        res.astype('float32').tofile(f"{dir}/epoches_000{epoch:03}/template{i}/template{i}.raw")
        for j in range(0,len(groups[i])):
            num = groups[i][j]
            data_block = data_block_array[num]
            data_block = data_block.reshape(-1,1)
            data_block = torch.Tensor(data_block).to(torch.float32).to(get_model_device(model))
            res = torch.Tensor(res).to(torch.float32).to(get_model_device(model))
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res, data_block)
            loss_running = loss.item()
            losses[num] = loss_running
    
    #存储template与对应blocks的loss
    filepath = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}', f'epoches_{epoch:03d}_template_losses.txt')
    print('store losses:',losses)
    with open(filepath,'wb') as file:
        pickle.dump(losses,file)
        
    

def save_blocks():
    print("save blocks")
    # 获取blockid对应数据
    args = config_parser.get_args()
    epoch=input("choose epoch:")
    epoch = int(epoch)
    num=input("choose template number range:")
    path = os.path.join(args.basedir, args.expname, 'data_block_array.raw')
    data_block_array=np.fromfile(f"{path}", dtype='float')
    print(data_block_array.shape)
    data_block_array = data_block_array.reshape(int(data_block_array.size/BLOCK_SIZE**3),BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)
    print(data_block_array.shape)

    # 获取groups的id
    path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}',f'epoches_{epoch:03}_groups.txt')
    print(path)
    with open(path,'rb') as file:
        groups = pickle.load(file)
    print(groups)
    
    # 存储templates对应block数据
    for i in range(0,int(num)):
        # 创建文件夹
        path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}', f'template{i}')
        os.makedirs(path, exist_ok=True)
        block_path = os.path.join(path, 'blocks')
        os.makedirs(block_path, exist_ok=True)
        # 输出每个template对应block原始数据至文件夹block_path
        for j in range(0,len(groups[i])):
            num = groups[i][j]
            filepath = os.path.join(block_path, f'block{num}.raw')
            data_block = data_block_array[num]
            data_block.astype('float32').tofile(f'{filepath}')
        

def save_fit_block():
    #加载神经网络
    args = config_parser.get_args()
    model, network_query_fn = networks.create_siren_maml(args)
    print("save fit blocks")
    epoch=input("choose template epoch:")
    epoch = int(epoch)
    num=input("choose template number range:")

    # 载入groups
    path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}',f'epoches_{epoch:03}_groups.txt')
    with open(path,'rb') as file:
        groups = pickle.load(file)


    #获取data_block_array
    path = os.path.join(args.basedir, args.expname, 'data_block_array.raw')
    data_block_array=np.fromfile(f"{path}", dtype='float')
    data_block_array = data_block_array.reshape(32,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)
    

    #绘制fit template 并且保存loss
    path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}')
    for i in range(0,int(num)):
        # 载入网络参数
        template_path = os.path.join(path, f'000{i:03}_template.tar')
        model.load_state_dict(torch.load(template_path)['network_fn_state_dict'])

        # 绘制fit blocks
        # 创建文件夹
        group_path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}', f'template{i}')
        os.makedirs(group_path, exist_ok=True)
        fit_block_path = os.path.join(group_path, 'fit_blocks')
        os.makedirs(fit_block_path, exist_ok=True)
       
        # 绘制拟合数据
        for j in range(0,len(groups[i])):
            # 拿出训练数据
            print(groups[i])
            data = []
            block_i = Block(data_block_array[groups[i][j]], np.array([BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE]))
            data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                    shuffle=True, num_workers=2)
            for batch_x, batch_y in data_loader:
                data.append([batch_x, batch_y])
            
            # 拟合template
            model_temp = deepcopy(model)
            maml.maml_optimize_template(model_temp, data, network_query_fn, args.query_steps, lrate=args.query_lrate)
            
            #绘制存储
            data_block_size = vec3f(BLOCK_SIZE)
            res = template_to_volume([model_temp,network_query_fn], data_block_size)
            num = groups[i][j]
            filepath = os.path.join(fit_block_path, f'block{num}_fit.raw')
            np.array(res).astype('float32').tofile(f"{filepath}")
        

def render_result():
    # 输入参数
    args = config_parser.get_args()
    epoch=input("choose epoch:")
    epoch = int(epoch)
    num=input("choose template number range:")

    # 读入拟合block以及template的losses 同时读入groups
    path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}',f'epoches_{epoch:03}_losses.txt')
    with open(path,'rb') as file:
        losses = pickle.load(file)
    path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}',f'epoches_{epoch:03}_template_losses.txt')
    with open(path,'rb') as file:
        template_losses = pickle.load(file)
    path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}',f'epoches_{epoch:03}_groups.txt')
    with open(path,'rb') as file:
        groups = pickle.load(file)


    # 绘制blocks
    path = os.path.join(args.basedir, args.expname, f'epoches_{epoch:06}')
    for i in range(0,int(num)):
        template_path = os.path.join(path, f'template{i}')
        # blocks=[]
        # for filepath in os.listdir(f'{template_path}'):
        #     block = np.fromfile(filepath, dtype=float)
        #     block.reshape(50,50,50)
        #     blocks.append(block)
        # 获取绘制文件列表
        filelist = []
        if len(os.listdir(os.path.join(template_path, 'blocks')))==0:
            continue
        index = random.randint(0,len(os.listdir(os.path.join(template_path, 'blocks')))-1)
        file1=os.listdir(os.path.join(template_path, 'blocks'))[index]
        file2=os.listdir(os.path.join(template_path, 'fit_blocks'))[index]
        filelist.append(os.path.join(template_path, 'blocks', file1))
        filelist.append(os.path.join(template_path, 'fit_blocks', file2))
        filelist.append(os.path.join(template_path, f'template{i}.raw'))
        print(filelist)

        # 获取文件对应loss
        index1 = groups[i][index]
        loss = losses[index1]
        template_loss = template_losses[index1]
        print(f"block{index1} fit loss:", loss, f"block{index1} template loss:", template_loss)

        ren = VolumeRender(filelist, [], [BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE], 3)
        ren.render()

    
def main():
    # save_blocks()
    # save_templates()
    # save_fit_block()
    render_result()

if __name__ == '__main__':
    main()

