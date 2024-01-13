from utils import *
from networks import create_mlp, create_siren, create_film_siren
import torch
import torch.nn as nn
import torch.nn.parallel
from config_parser import get_args
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,ALL_COMPLETED
from maml import maml_optimize_template
from copy import deepcopy


def inference(model, fn, inputs):
    return fn(inputs, model)


# TODO 修改成AdaptiveMultiMLP(需要降低输入网络的points数量，防止显存爆炸)
class MultiNetwork(nn.Module):
    def __init__(self, args, num_networks,
                 use_single_net=False, use_same_initialization_for_all_networks=False,
                 network_rng_seed=False, weight_initialization_method='', bias_initialization_method='standard'):
        super().__init__()
        self.num_networks = num_networks
        self.output_ch = args.output_ch
        self.use_single_net = use_single_net
        self.use_same_initialization_for_all_net_work = use_same_initialization_for_all_networks
        self.network_rng_seed = network_rng_seed
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        # start_iter, network_query_fn, grad_vars, optimizer, model组成的tuple的数组
        if args.model == 'mlp_relu':
            self.multi_MLP = [create_mlp(args) for _ in range(self.num_networks)]
        elif args.model == 'siren':
            self.multi_MLP = [create_siren(args) for _ in range(self.num_networks)]
        elif args.model == 'film_siren':
            self.multi_MLP = [create_film_siren(args) for _ in range(self.num_networks)]
        else:
            raise NotImplementedError
        # self.pool = ThreadPoolExecutor(max_workers=3)
        # TODO: 初始化多个小网络的参数

    # 根据batch_size 训练的时候, 输入的x要整理成[num_networks ,batch, input_ch] 要减小batch的大小
    def forward(self, x):
        # # return self.forward_instant(x)
        # coord_index = discretize_domain_to_grid(self.domain_min, self.domain_max, self.res, x)
        # # normalize x
        # coord_index = coord_index.int()
        # # 拷贝了一次
        # x = normalize_point_in_grid(x, self.domain_min, self.domain_max, self.res)
        # result = torch.zeros([len(coord_index), 1], dtype=torch.float32, requires_grad=True).cuda()
        #
        # for index in range(self.num_networks):
        #     result[index * batch_size_per_network:(index + 1) * batch_size_per_network, :] \
        #         = self.multi_MLP[index][1](x[index * batch_size_per_network:(index + 1) * batch_size_per_network, :]
        #                                    , self.multi_MLP[index][-1])
        # return result
        result = torch.zeros([x.shape[0], x.shape[1], self.output_ch], dtype=torch.float32, requires_grad=True).cuda()
        for i in range(self.num_networks):
            result[i, ...] = self.multi_MLP[i][1](x[i, ...], self.multi_MLP[i][-1])
        # 多线程
        # task = []
        # for i in range(self.num_networks):
        #     task.append(self.pool.submit(inference, self.multi_MLP[i][-1], self.multi_MLP[i][1], x[i, ...]))
        # res = wait(task, return_when=ALL_COMPLETED)
        # for i, r in enumerate(res.done):
        #     result[i, ...] = r.result()
        return result

    def optimizer_step(self):
        for MLP in self.multi_MLP:
            MLP[3].step()

    def optimizer_zero_grad(self):
        for MLP in self.multi_MLP:
            MLP[3].zero_grad()

    # 效率很低
    def initialize_multi_mlp(self, templates, data_block_loader, query_steps=3, query_lrate=1e-3):
        # 训练数据按批次拿出来
        data = [[] for i in range(self.num_networks)]
        for batch_x, batch_y in data_block_loader:
            batch_x = np.squeeze(batch_x, axis=0)
            batch_y = np.squeeze(batch_y, axis=0)
            for i in range(self.num_networks):
                data[i].append([batch_x[i, ...], batch_y[i, ...]])
        loss_average = 0.0
        for i in range(self.num_networks):
            loss = []
            for [template, fn] in templates:
                template_copy = deepcopy(template)
                loss_i_j = maml_optimize_template(template_copy, data[i], fn, optimize_steps=query_steps,
                                                  lrate=query_lrate)
                del template_copy
                loss.append(loss_i_j)
            # 得到最小的template
            min_j = np.argmin(np.array(loss))
            # 替换model的参数
            self.multi_MLP[i][-1].load_state_dict(templates[min_j][0].state_dict())
            # loss_average += np.min(np.array(loss))
        #     print(np.max(np.array(loss)), np.min(np.array(loss)))
        # print(loss_average/self.num_networks)

    def decay_lrate(self, new_lrate):
        for MLP in self.multi_MLP:
            optimizer = MLP[3]
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

    # 在path里面指定epoch
    def saved_checkpoints(self, path):
        i = 0
        for MLP in self.multi_MLP:
            path_ = os.path.join(path, '{:06d}.tar'.format(i))
            torch.save({
                'network_fn_state_dict': MLP[-1].state_dict(),
            }, path_)
            print('Saved checkpoints at', path_)
            i += 1

    def load_checkpoints(self, path):
        pass


# semi-fast querying 将整个输入的points(随机采样的数据直接学习) 此步用于推理
# TODO 修改成现有 AdaptiveMultiMLP
def query_multi_network(multi_network, domain_min, domain_max, points,
                        occupancy_grid, res):
    # scale to [-1, 1] 效率
    global_domain_size = domain_max - domain_min
    fixed_resolution = torch.tensor(res, dtype=torch.long, device=points.device)
    network_strides = torch.tensor([res[2] * res[1], res[0], 1], dtype=torch.long,
                                   device=points.device)  # assumes row major ordering
    voxel_size = global_domain_size / fixed_resolution
    point_indices_3d = ((points - domain_min) / voxel_size).to(network_strides)
    point_indices = (point_indices_3d * network_strides).sum(dim=1)
    num_networks = multi_network.num_networks

    if occupancy_grid is not None:
        occupancy_resolution = torch.tensor(res, dtype=torch.long, device=points.device)
        strides = torch.tensor([res[2] * res[1], res[2], 1], dtype=torch.long,
                               device=points.device)  # assumes row major ordering
        voxel_size = global_domain_size / occupancy_resolution
        occupancy_indices = ((points - domain_min) / voxel_size).to(torch.long)
        torch.max(torch.tensor([0, 0, 0], device=points.device), occupancy_indices, out=occupancy_indices)
        torch.min(occupancy_resolution - 1, occupancy_indices, out=occupancy_indices)
        occupancy_indices = (occupancy_indices * strides).sum(dim=1)

        point_in_occupied_space = occupancy_grid[occupancy_indices]
        del occupancy_indices

    # Filtering points outside global domain
    epsilon = 0.001
    active_samples_mask = torch.logical_and((points > domain_min + epsilon).all(dim=1),
                                            (points < domain_max - epsilon).all(dim=1))
    if occupancy_grid is not None:
        active_samples_mask = torch.logical_and(active_samples_mask, point_in_occupied_space)
        del point_in_occupied_space
    proper_index = torch.logical_and(point_indices >= 0,
                                     point_indices < num_networks)
    active_samples_mask = torch.nonzero(torch.logical_and(active_samples_mask, proper_index), as_tuple=False).squeeze()
    del proper_index

    filtered_point_indices = point_indices[active_samples_mask]
    del point_indices

    # Sort network
    filtered_point_indices, reorder_indices = torch.sort(filtered_point_indices)

    # make sure that also batch sizes are given for networks which are queried 0 points
    contained_nets, batch_size_per_network_incomplete = torch.unique_consecutive(filtered_point_indices,
                                                                                 return_counts=True)
    del filtered_point_indices
    batch_size_per_network = torch.zeros(num_networks, device=points.device, dtype=torch.long)
    batch_size_per_network[contained_nets] = batch_size_per_network_incomplete
    batch_size_per_network = batch_size_per_network.cpu()

    points_reordered = points[active_samples_mask]

    # reorder so that points handled by the same network are packed together in the list of points
    points_reordered = points_reordered[reorder_indices]

    num_points_to_process = points_reordered.size(0) if points_reordered.ndim > 0 else 0
    print("#points to process:", num_points_to_process, flush=True)

    if num_points_to_process == 0:
        return torch.zeros(1, points.size(0), dtype=torch.float, device=points_reordered.device)

    # Convert global to local coordinates
    points_reordered = normalize_point_in_grid(points_reordered)

    return multi_network(points_reordered, batch_size_per_network)


# teacher-student model的训练数据生成 training_data_per_grid 指的是每个格子里面的随机产生的点的个数
# 函数返回应该是all_examples, all_results
def generate_training_data(mlp, domain_min, domain_max, res, training_data_per_grid):
    # 产生每个小格子的随机点 格式为 res[0],res[1],res[2],2,3
    with torch.no_grad():
        aabb_grid = discretize_domain_to_aabb(domain_min, domain_max, res)
        all_examples = torch.zeros([vec3_len(res) * training_data_per_grid, 3]).cuda()
        all_results = torch.zeros([vec3_len(res) * training_data_per_grid, 1]).cuda()
        for i in range(res[0]):
            for j in range(res[1]):
                for k in range(res[2]):
                    all_examples[get_index_from_xyz(i, j, k, res) * training_data_per_grid:(get_index_from_xyz(i, j, k,
                                                                                                               res)
                    + 1) * training_data_per_grid, :] = \
                        get_random_points_inside_domain(training_data_per_grid, aabb_grid[i, j, k, 0, :]
                                                        , aabb_grid[i, j, k, 1, :])
                    all_results[get_index_from_xyz(i, j, k, res) * training_data_per_grid:(get_index_from_xyz(i, j, k,
                                                                                                              res)
                    + 1) * training_data_per_grid, :] = mlp(all_examples[get_index_from_xyz(i, j, k, res) *
                                                            training_data_per_grid:(get_index_from_xyz(i, j, k, res)
                                                            + 1) * training_data_per_grid, :])
    return all_examples, all_results


# teacher-student model 输入随机产生好的数据(按照块产生),训练子网络
def train_multi_network(multi_network, all_examples, all_results, domain_min, domain_max, batch_size, res, args):
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    res_fn = multi_network(all_examples, batch_size)
    # res_fn = multi_network.forward_instant(normalize_point_in_grid(all_examples, domain_min.to(get_device(args.GPU)),
    #                        domain_max.to(get_device(args.GPU)), res.to(get_device(args.GPU))))
    train_loss = MSE_loss(res_fn, all_results)
    print(train_loss)
    multi_network.optimizer_zero_grad()
    train_loss.backward()
    multi_network.optimizer_step()


# 单例测试 全部转换成小网络后
# baseline
def main():
    args = get_args()
    res = vec3f(10)
    data_size = 200 * 200 * 200
    batch_size = data_size // vec3_len(res)
    all_examples = get_random_points_inside_domain(batch_size * vec3_len(res), domain_min=vec3f(0),
                                                   domain_max=vec3f(1000))
    all_examples = torch.from_numpy(all_examples).to(torch.float32).to(get_device(args.GPU))
    all_results = torch.randn([batch_size * vec3_len(res), 1]).to(get_device(args.GPU))
    nets = MultiNetwork(args, vec3_len(res), domain_min=vec3f(0), domain_max=vec3f(1000), res=res)
    for i in range(20000):
        print(i)
        train_multi_network(nets, all_examples, all_results, domain_min=vec3f(0),
                            domain_max=vec3f(1000), batch_size=batch_size, res=res, args=args)


if __name__ == '__main__':
    main()
