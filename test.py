from utils import *
from dataloaders import *
from losses import plot_curve
from tqdm import tqdm, trange
from datetime import datetime
from itertools import islice
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import networks
import config_parser
import MetaTemplate
import MI
import struct
import utils
import gc
import json


def calculate_blocks_entropy(data_block_array):
        block_entropy_calculator = MI.Variable_MI()
        max_entropy = 0
        index = -1
        for i, block in tqdm(enumerate(data_block_array), total=len(data_block_array)):
                temp_entropy = block_entropy_calculator.variable_entropy(block.v)
                if temp_entropy > max_entropy:
                        index = i
                        max_entropy = temp_entropy
        return index, max_entropy

def choose_similar_blocks(sample, data_block_array):
        block_JSD_calculator = MI.Variable_MI()
        similar_blocks_index = []
        for i, block in enumerate(tqdm(data_block_array)):
                jsd = block_JSD_calculator.jsd(sample.v, block.v)
                if jsd < 0.2:
                        similar_blocks_index.append(i)
        return similar_blocks_index

def calculate_PSNR(max,loss):
        return 10*math.log10((max**2)/loss)

def train_base_network(args, data_block_array, save=True):
        # preprocess
        # index, entropy = calculate_blocks_entropy(data_block_array)
        # print("index:", index, "entropy:", entropy, "min:", data_block_array[index].v.min(), "max:", data_block_array[index].v.max())
        # vtk_draw_blocks(data_block_array, off_screen=False)

        # load data
        dataset = MetaDataset(data_block_array, args.maml_chunk)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)

        # create network
        model, network_query_fn = networks.create_transformer_modulated_siren_maml(args)
        MSE_loss = torch.nn.MSELoss(reduction='mean')
        TemplateGenerator = MetaTemplate.MetaTemplate(model, network_query_fn, MSE_loss, args, meta_lr=args.meta_lr, update_lr=args.update_lr,
                                        num_meta_steps=args.meta_steps)

        # train
        steps = []
        for i in trange(0, args.maml_epoches):
                loss_running = 0.0
                steps.append(i)
                for j, data_i in enumerate(tqdm(data_loader)):
                        data_i = data_i[:, torch.randperm(data_i.size(1))]
                        set_ = meta_split(data_i, split_mode=args.meta_split_method)
                        x_spt, y_spt, x_qry, y_qry = set_['context'][0], set_['context'][1], set_['query'][0], set_['query'][1]
                        x_spt = np.concatenate((x_spt, np.ones(list(x_spt.shape[:-1])+[1])), axis=-1)
                        x_qry = np.concatenate((x_qry, np.ones(list(x_qry.shape[:-1])+[1])), axis=-1)
                        modulations = torch.zeros(args.batch_size, args.latent_dim).requires_grad_()
                        if torch.cuda.is_available():
                                x_spt = torch.from_numpy(x_spt).to(torch.float32).to(get_device(args.GPU))
                                y_spt = torch.from_numpy(y_spt).to(torch.float32).to(get_device(args.GPU))
                                x_qry = torch.from_numpy(x_qry).to(torch.float32).to(get_device(args.GPU))
                                y_qry = torch.from_numpy(y_qry).to(torch.float32).to(get_device(args.GPU))
                                modulations = modulations.to(torch.float32).to(get_device(args.GPU))
                        loss = TemplateGenerator.variant_forward(x_spt, y_spt, x_qry, y_qry, modulations, i, args.inner_part)
                        psnr = calculate_PSNR(y_spt.max(), loss)
                        tqdm.write(f"[MAML] Iter: {i} Loss: {loss}, PSNR: {psnr}")
                        loss_running += loss
                        # loss_running += TemplateGenerator(x_spt, y_spt, x_qry, y_qry)
                        torch.cuda.empty_cache()
                        
                loss_running /= len(data_loader)
                psnr = calculate_PSNR(dataset.getmax(), loss_running)
                tqdm.write(f"[MAML] Iter: {i} Loss: {loss_running}, PSNR: {psnr}")
                if i%500==0 and i!=0 and save:
                        # save model and config
                        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        os.makedirs(os.path.join(args.basedir, args.expname, time), exist_ok=True)
                        model_path = os.path.join(args.basedir, args.expname, time, f"{args.task}_iter{i}.tar")
                        torch.save({'network_fn_state_dict': model.state_dict(),}, model_path)
                        config_path = os.path.join(args.basedir, args.expname, time, "config.txt")
                        args_dict = vars(args)
                        with open(config_path, 'w') as config_file:
                                json.dump(args_dict, config_file, indent=4)
                if loss_running < 1e-4:
                        break
        return psnr

def test_base_network(args, data_block_array):
        # load data
        dataset = MetaDataset(data_block_array, args.maml_chunk)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                shuffle=False, num_workers=2)

        # load network
        model, network_query_fn = networks.create_transformer_modulated_siren_maml(args)
        state_dict = torch.load('/home/XiYang/MAML_KiloNet/source/logs/for_debug/2024-01-11 19:39:41/train_templates_multi_timestamp_iter2500.tar')
        state_dict = OrderedDict(islice(state_dict['network_fn_state_dict'].items(), 0, 14))
        current_state_dict = model.state_dict()
        current_state_dict.update(state_dict)
        model.load_state_dict(current_state_dict)
        MSE_loss = torch.nn.MSELoss(reduction='mean')
        TemplateGenerator = MetaTemplate.MetaTemplate(model, network_query_fn, MSE_loss, args, meta_lr=args.meta_lr, update_lr=args.update_lr,
                                        num_meta_steps=args.meta_steps)
        

        # evaluate model
        hard_blocks = []
        losses = []
        for k, data_i in enumerate(tqdm(data_loader)):
                data_i = data_i[:, torch.randperm(data_i.size(1))]
                set_ = meta_split(data_i, split_mode=args.meta_split_method)
                x_spt, y_spt, x_qry, y_qry = set_['context'][0], set_['context'][1], set_['query'][0], set_['query'][1]
                x_spt = np.concatenate((x_spt, np.ones(list(x_spt.shape[:-1])+[1])), axis=-1)
                x_qry = np.concatenate((x_qry, np.ones(list(x_qry.shape[:-1])+[1])), axis=-1)
                modulations = torch.zeros(1, args.latent_dim).requires_grad_()
                if torch.cuda.is_available():
                        x_spt = torch.from_numpy(x_spt).to(torch.float32).to(get_device(args.GPU))
                        y_spt = torch.from_numpy(y_spt).to(torch.float32).to(get_device(args.GPU))
                        x_qry = torch.from_numpy(x_qry).to(torch.float32).to(get_device(args.GPU))
                        y_qry = torch.from_numpy(y_qry).to(torch.float32).to(get_device(args.GPU))
                        modulations = modulations.to(torch.float32).to(get_device(args.GPU))
                MTF_parameters = TemplateGenerator.modulation_transformer_forward(x_spt, y_spt, x_qry, y_qry)
                for j in trange(0, args.query_steps):
                        # modulations = TemplateGenerator.modulation_forward(x_spt, y_spt, x_qry, y_qry, modulations)
                        MTF_parameters = TemplateGenerator.modulation_transformer_forward(x_spt, y_spt, x_qry, y_qry, MTF_parameters[0])
                # prediction = model.modulated_forward(x_spt, modulations[0])
                        prediction = model.MTF_forward(x_spt, MTF_parameters[0])
                        loss_running = MSE_loss(prediction, y_spt).item()
                        psnr = calculate_PSNR(y_spt.max(),loss_running)
                        tqdm.write(f"Block: {k} Loss: {loss_running}, PSNR: {psnr}")
                losses.append(loss_running)
                if psnr < 25:
                        hard_blocks.append(k)
                
                del x_spt, y_spt, x_qry, y_qry, prediction, modulations, loss_running
                gc.collect()
                torch.cuda.empty_cache()

        return hard_blocks, losses

def erase_blocks(whole_volume, index_array, size, res):
        w,h,d = np.array(res) // size
        for index in range(w*h*d):
                if index not in index_array:
                        k = int(index / (w*h))
                        j = int((index % (w*h)) / w)
                        i = int((index % (w*h)) % w)
                        whole_volume.v[i*size:(i+1)*size,
                                        j*size:(j+1)*size,
                                        k*size:(k+1)*size]=0
        # whole_volume.v[100:600,]

def main():
        args = config_parser.get_args()
        # select dataset
        if args.dataset == "Argon_Bubble":
                Dataset = ArBubbleDataset('Argon-Bubble')
        elif args.dataset == "asteroid":
                Dataset = MultiTimeStampDataset('asteroid', 'v02')

        # read data
        Dataset.read_data() 
        block_generator = BlockGenerator(Dataset.get_scalar_data(), Dataset.get_data_res())
        data_block_array = block_generator.generate_data_block(args.block_size, method=args.block_gen_method, block_num=args.block_num)
        whole_volume = Block(Dataset.get_scalar_data()[0], [640,256,256], pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), [640,256,256]).reshape([-1, 3]))
        vtk_draw_blocks([whole_volume])
        
        if args.task == "train_templates_multi_timestamp":
                block_index = [0]
                erase_blocks(whole_volume, block_index, args.block_size[0], [640,256,256])
                vtk_draw_blocks([whole_volume])
                train_base_network(args, [data_block_array[21]])
        elif args.task == "test_network":
                hard_blocks, losses = test_base_network(args, data_block_array)
                average_loss = sum(losses)/len(losses)
                print(f'average loss: {average_loss}, PSNR: {-10*math.log10(average_loss)}')
                print('blocks index:', hard_blocks, 'blocks length:', len(hard_blocks))
        elif args.task == "test_bad_blocks":
                # bad_blocks_index = [5, 6, 7, 17, 18, 19, 28, 29, 30, 31, 41, 42, 43, 53, 54, 55, 66, 78, 79, 89, 90, 91, 114, 115, 124, 125, 126, 127, 137, 138, 139, 148, 149, 150, 151, 161, 162, 163, 172, 173, 174, 175, 185, 186, 197, 198, 199, 209, 210, 211, 221, 222, 233, 234, 235, 245, 246, 247, 258, 259, 269, 270, 271, 281, 282, 283, 293, 294, 295]
                # bad_blocks_index = [4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 28, 29, 30, 31, 32, 40, 41, 42, 43, 44, 52, 53, 54, 55, 56, 65, 66, 67, 76, 77, 78, 79, 80, 89, 90, 91, 100, 101, 102, 103, 113, 114, 115, 124, 125, 126, 127, 128, 136, 137, 138, 139, 140, 148, 149, 150, 151, 152, 160, 161, 162, 163, 164, 172, 173, 174, 175, 176, 184, 185, 186, 187, 188, 196, 197, 198, 199, 208, 209, 210, 211, 212, 220, 221, 222, 223, 232, 233, 234, 235, 236, 244, 245, 246, 247, 256, 257, 258, 259, 260, 268, 269, 270, 271, 280, 281, 282, 283, 284, 292, 293, 294, 295, 296]
                # erase_blocks(whole_volume, bad_blocks_index, args.block_size[0], [640,256,256])
                # vtk_draw_blocks([whole_volume])
                
                # psnrs = []
                # for index in bad_blocks_index:
                #         _, entropy = calculate_blocks_entropy([data_block_array[index]])
                #         print("index:", index, "entropy:", entropy)
                #         psnrs.append(train_base_network(args, [data_block_array[index]], save=False))
                #         test_base_network(args, [data_block_array[index]])
                # print(psnrs)

                block1 = Block(whole_volume.v[315:365, 125:175, 125:175], [50,50,50], pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), [50,50,50]).reshape([-1, 3]))
                block2 = Block(whole_volume.v[335:385, 145:195, 145:195], [50,50,50], pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), [50,50,50]).reshape([-1, 3]))
                block3 = Block(whole_volume.v[345:395, 135:185, 135:185], [50,50,50], pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), [50,50,50]).reshape([-1, 3]))
                block4 = Block(whole_volume.v[310:390, 110:190, 110:190], [80,80,80], pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), [80,80,80]).reshape([-1, 3]))
                hard_block_array = [block4]
                test_base_network(args, hard_block_array)
        elif args.task == "test_similar_blocks":
                similar_blocks_index = choose_similar_blocks(data_block_array[150], data_block_array)
                print("index:", similar_blocks_index, 'number:', len(similar_blocks_index))
                erase_blocks(whole_volume, similar_blocks_index, args.block_size[0], [640,256,256])
                vtk_draw_blocks([whole_volume])

                similar_block_array = [data_block_array[index] for index in similar_blocks_index]
                train_base_network(args, similar_block_array)

if __name__ == '__main__':
    main()
