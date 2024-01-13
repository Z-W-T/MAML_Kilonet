import numpy as np
from torch.utils.data import Dataset, DataLoader
import vtk
from glob import glob
from vtkmodules.util import numpy_support
import netCDF4 as nc
from utils import *
import struct

def output_meta_dataset(context_inputs, context_targets, test_inputs, test_targets):
    context_inputs = np.stack(context_inputs, axis=0)
    context_targets = np.stack(context_targets, axis=0)
    test_inputs = np.stack(test_inputs, axis=0)
    test_targets = np.stack(test_targets, axis=0)
    meta_data = {'context': (context_inputs, context_targets),
                 'query': (test_inputs, test_targets)}
    return meta_data


# support和query 划分
def meta_split(volume_tensor, split_mode='all', ratio=0.8):
    context_inputs = []
    context_targets = []
    test_inputs = []
    test_targets = []
    # [task_num, batch, 4]
    batch_size = volume_tensor.shape[0]

    if split_mode is 'partial':
        # Subsample half of the points as context
        for b in range(batch_size):
            idx = torch.randperm(volume_tensor[b].shape[0])  # shuffle along points dimension
            # shuffle block 内部的数据
            volume_tensor[b] = volume_tensor[b][idx]
            context_length = int(np.floor(volume_tensor[b].shape[0] * ratio))
            context_inputs.append(volume_tensor[b][:context_length, :3])
            context_targets.append(volume_tensor[b][:context_length, 3:])
            test_inputs.append(volume_tensor[b][context_length:, :3])
            test_targets.append(volume_tensor[b][context_length:, 3:])
        return output_meta_dataset(context_inputs, context_targets, test_inputs, test_targets)
    elif split_mode is 'all':
        for b in range(batch_size):
            context_inputs.append(volume_tensor[b][:, :3])
            context_targets.append(volume_tensor[b][:, 3:])
            test_inputs.append(volume_tensor[b][:, :3])
            test_targets.append(volume_tensor[b][:, 3:])
        return output_meta_dataset(context_inputs, context_targets, test_inputs, test_targets)
    elif split_mode is 'random':
        # TODO 设置随机 meta_tensor
        pass
    else:
        raise NotImplementedError


# volume数据的接口(包括) 二进制文件
class VolumeData:
    def __init__(self, file_path, w, h, d, args):
        super(VolumeData, self).__init__()
        self.file_path = file_path
        self.w = w
        self.h = h
        self.d = d
        self.args = args
        self.is_load = False
        self.data = None    # 数据的存储接口

    def load_gpu(self):
        # 将数据直接load torch cuda
        pass

    def free_gpu(self):
        pass

    # 给一个中心点和采样半径,按块进行随机撒点采样
    def sample_data_block(self, center, sample_radius, sample_method=None):
        pass

    # 根据任务进行采样
    def sample_data_task(self, task, sample_method=None):
        # 调用sample_data_block
        pass


class Block(Dataset):
    # TODO 降低内存消耗 利于多个数据块的训练，可能cache也会更好一些
    def __init__(self, block_volume, block_size, pos, chunk=1024*256):
        self.v = block_volume
        self.res = np.array(block_size)
        self.chunk = chunk
        self.pos = pos

    def __getitem__(self, item):
        # 直接取值
        # point = index_to_domain_xyz(item, vec3f(0.0), vec3f(1.0), self.res)
        # xyz = index_to_domain_xyz_index(item, self.res)
        # val = self.v[xyz[0], xyz[1], xyz[2]]
        # return point, np.array([val])

        pos_ = self.pos[item*self.chunk:(item+1)*self.chunk, ...]
        vol = self.v.reshape([-1,1])
        vol = vol[item*self.chunk:(item+1)*self.chunk, ...]
        num = generate_shuffle_number(vol.shape[0])
        # shuffle
        return pos_[num, ...], vol[num, ...]

    def __len__(self):
        num_points = self.res.prod()
        return int(np.ceil(num_points/self.chunk))


class BlockV2(Dataset):
    # TODO 降低内存消耗 利于多个数据块的训练，可能cache也会更好一些
    def __init__(self, volume_dataset, left, right, block_size, pos, chunk=1024*256, timestamp=None):
        self.v = volume_dataset
        self.left = left
        self.right = right
        self.res = block_size
        self.chunk = chunk
        self.stamp = timestamp
        self.pos = pos

    def __getitem__(self, item):
        # 直接取值
        # point = index_to_domain_xyz(item, vec3f(0.0), vec3f(1.0), self.res)
        # xyz = index_to_domain_xyz_index(item, self.res)
        # val = self.v[xyz[0], xyz[1], xyz[2]]
        # return point, np.array([val])

        pos_ = self.pos[item*self.chunk:(item+1)*self.chunk, ...]
        if self.stamp is None:
            vol = self.v[self.left[0]:self.right[0],
                         self.left[1]:self.right[1],
                         self.left[2]:self.right[2]].reshape([-1,1])
        else:
            vol = self.v[self.stamp,
                  self.left[0]:self.right[0],
                  self.left[1]:self.right[1],
                  self.left[2]:self.right[2]].reshape([-1, 1])
        vol = vol[item*self.chunk:(item+1)*self.chunk, ...]
        num = generate_shuffle_number(vol.shape[0])
        # shuffle
        return pos_[num, ...], vol[num, ...]

    def __len__(self):
        num_points = self.res.prod()
        return int(np.ceil(num_points/self.chunk))


class BlockSum(Dataset):
    def __init__(self, block_array):
        self.block_array = block_array
        self. num = len(block_array)

    def __getitem__(self, item):
        output_pos = []
        output_vol = []
        for block in self.block_array:
            pos, vol = block[item]
            output_pos.append(pos)
            output_vol.append(vol)
        return np.array(output_pos), np.array(output_vol)

    def __len__(self):
        return len(self.block_array[0])


# block_generator 产生 block 的数据
class BlockGenerator:
    def __init__(self, volume_data, res, chunk=1024*512):
        super(BlockGenerator, self).__init__()
        self.res = np.array(res)
        # 可能是四维数组或者三维数组
        self.volume_data = volume_data
        if self.volume_data.ndim == 4:
            self.has_timestamp = True
        else:
            self.has_timestamp = False
        self.chunk = chunk

    # res为volume的大小, size为block的大小
    def uniform_part(self, size):
        pos = get_query_coords(vec3f(-1), vec3f(1), size).reshape([-1, 3])
        data_block_array = []
        if self.has_timestamp is False:
            [w, h, d] = self.res // size
            delta = size
            for k in range(d):
                for j in range(h):
                    for i in range(w):
                        data_block = self.volume_data[i*delta[0]:(i+1)*delta[0],
                                                      j*delta[1]:(j+1)*delta[1],
                                                      k*delta[2]:(k+1)*delta[2]]
                        data_block = Block(data_block, size, pos, self.chunk)
                        data_block_array.append(data_block)
        else:
            [w, h, d] = self.res[1:] // size
            t = self.res[0]
            delta = size
            for time in range(t):
                for k in range(d):
                    for j in range(h):
                        for i in range(w):
                            data_block = self.volume_data[time,
                                         i * delta[0]:(i + 1) * delta[0],
                                         j * delta[1]:(j + 1) * delta[1],
                                         k * delta[2]:(k + 1) * delta[2]]
                            data_block = Block(data_block, size, pos, self.chunk)
                            data_block_array.append(data_block)

        return data_block_array

    # return num 个 block
    # TODO 此处可能存在问题，可能是随机中心点，在概率上更加正确
    def random_sample(self, size, block_num=None):
        pos = get_query_coords(vec3f(-1), vec3f(1), size).reshape([-1, 3])
        data_block_array = []
        if self.has_timestamp is False:
            if block_num is None:
                [w, h, d] = self.res // size
                block_num = w*h*d
            # 此处有bug
            num = (self.res - size).prod()
            xyz = generate_shuffle_number(num)

            for i in range(block_num):
                left = index_to_domain_xyz_index(xyz[i], self.res-size)
                right = left + size
                data_block = self.volume_data[left[0]:right[0],
                                              left[1]:right[1],
                                              left[2]:right[2]]
                data_block = Block(data_block, size, pos, self.chunk)
                data_block_array.append(data_block)
        else:
            times = self.res[0]
            if block_num is None:
                [w, h, d] = self.res[1:] // size
                block_num = w * h * d
            num = (self.res[1:] - size).prod()
            for time in range(times):
                xyz = generate_shuffle_number(num)
                for i in range(block_num):
                    left = index_to_domain_xyz_index(xyz[i], self.res[1:] - size)
                    right = left + size
                    data_block = self.volume_data[time,
                                                 left[0]:right[0],
                                                 left[1]:right[1],
                                                 left[2]:right[2]]
                    data_block = Block(data_block, size, pos, self.chunk)
                    data_block_array.append(data_block)
        return data_block_array

    # 考虑block_size也是随机的
    def generate_data_block(self, block_size, method='uniform', block_num=None):
        block_size = np.array(block_size)
        if method == 'uniform':
            return self.uniform_part(block_size)
        elif method == 'random':
            return self.random_sample(block_size, block_num)
        else:
            raise NotImplementedError

    def generate_data_block_in_center(self, block_size, distance):
        pos = get_query_coords(vec3f(-1), vec3f(1), block_size).reshape([-1, 3])
        data_block_array = []
        center = np.array([self.res[0]//2, self.res[1]//2, self.res[2]//2])
        t = np.array([[0,0,0],[-1,-1,-1],[-1,-1,1],[-1,1,1],
             [1,-1,1],[1,1,-1],[-1,1,-1],[1,-1,-1],[1,1,1]])
        for i in range(9):
            center_ = center + t[i]*distance
            data_block = self.volume_data[center_[0]-block_size[0]//2:center_[0]+block_size[0]//2,
                         center_[1]-block_size[1]//2:center_[1]+block_size[1]//2,
                         center_[2]-block_size[2]//2:center_[2]+block_size[2]//2]
            data_block = Block(data_block, block_size, pos, self.chunk)
            data_block_array.append(data_block)

        return data_block_array

    def generate_data_block_with_offset(self, block_size, num):
        pos = get_query_coords(vec3f(-1), vec3f(1), block_size).reshape([-1, 3])
        data_block_array = []
        center = np.array([self.res[0]//2-block_size[0]//2, self.res[1]//2-block_size[1]//2, self.res[2]//2-block_size[2]//2])
        t = np.random.random((num, 3))
        t = t-[0.5,0.5,0.5]
        print(t)
        for i in range(num):
            offset = t[i]*block_size[0]//2
            data_block = self.volume_data[int(center[0]+offset[0]):int(center[0]+offset[0]+block_size[0]),
                         int(center[1]+offset[1]):int(center[1]+offset[1]+block_size[1]),
                         int(center[2]+offset[2]):int(center[2]+offset[2]+block_size[2])]
            data_block = Block(data_block, block_size, pos, self.chunk)
            data_block_array.append(data_block)

        return data_block_array


# block_generator 产生 block 的数据
class BlockGeneratorV2:
    def __init__(self, volume_data, res, chunk=1024*512):
        super(BlockGeneratorV2, self).__init__()
        self.res = res
        # 可能是四维数组或者三维数组
        self.volume_data = volume_data
        if self.volume_data.ndim == 4:
            self.has_timestamp = True
        else:
            self.has_timestamp = False
        self.chunk = chunk
        self.pos = None

    # res为volume的大小, size为block的大小
    def uniform_part(self, size):
        data_block_array = []
        if self.has_timestamp is False:
            [w, h, d] = self.res // size
            delta = size
            for k in range(d):
                for j in range(h):
                    for i in range(w):
                        data_block = BlockV2(self.volume_data, np.array([i * delta[0], j*delta[1], k*delta[2]]),
                                             np.array([(i+1) * delta[0], (j+1)*delta[1], (k+1)*delta[2]]),
                                             size, self.pos, self.chunk)
                        data_block_array.append(data_block)
        else:
            [w, h, d] = self.res[1:] // size
            t = self.res[0]
            delta = size
            for time in range(t):
                for k in range(d):
                    for j in range(h):
                        for i in range(w):
                            data_block = self.volume_data[time,
                                         i * delta[0]:(i + 1) * delta[0],
                                         j * delta[1]:(j + 1) * delta[1],
                                         k * delta[2]:(k + 1) * delta[2]]
                            data_block = Block(data_block, size, self.chunk)
                            data_block_array.append(data_block)

        return data_block_array

    # return num 个 block
    # TODO 此处可能存在问题，可能是随机中心点，在概率上更加正确
    def random_sample(self, size, block_num=None):
        self.pos = get_query_coords(vec3f(-1), vec3f(1), size).reshape([-1, 3])
        data_block_array = []
        if self.has_timestamp is False:
            if block_num is None:
                [w, h, d] = self.res // size
                block_num = w*h*d
            # 此处有bug
            num = (self.res - size).prod()
            xyz = generate_shuffle_number(num)

            for i in range(block_num):
                left = index_to_domain_xyz_index(xyz[i], self.res-size)
                right = left + size
                data_block = BlockV2(self.volume_data, left, right, size, self.pos, self.chunk)
                data_block_array.append(data_block)
        else:
            times = self.res[0]

            if block_num is None:
                [w, h, d] = self.res[1:] // size
                block_num = w * h * d
            num = (self.res[1:] - size).prod()
            for time in range(times):
                xyz = generate_shuffle_number(num)
                for i in range(block_num):
                    left = index_to_domain_xyz_index(xyz[i], self.res[1:] - size)
                    right = left + size
                    data_block = BlockV2(self.volume_data, left, right, size, self.pos, self.chunk, time)
                    data_block_array.append(data_block)
        return data_block_array

    # 考虑block_size也是随机的
    def generate_data_block(self, block_size, method='uniform', block_num=None):
        block_size = np.array(block_size)
        if method == 'uniform':
            return self.uniform_part(block_size)
        elif method == 'random':
            return self.random_sample(block_size, block_num)
        else:
            raise NotImplementedError


# TODO
class MetaSplitter(Dataset):
    # Class for representing Chunked data of a volume data
    def __init__(self, meta_dataset, dense=True, mode='train', num_samples=64**2//2):
        super(MetaSplitter, self).__init__()
        self.dataset = meta_dataset
        self.dense = dense
        self.mode = mode
        self.num_samples = num_samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        pass


class AsteroidDataset(Dataset):
    def __init__(self, file_path, name, batch_points_num=1024*2, datatype='PointData', timestamp=None):
        super(AsteroidDataset, self).__init__()
        self.file_path = file_path
        self.name = name
        self.datatype = datatype
        self.timestamp = timestamp
        self.res = None
        self.v = None
        self.batch_points_num = batch_points_num

    def read_volume_data(self):
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(self.file_path)
        reader.Update()
        vtkimage = reader.GetOutput()

        print(
            "using" + " volume data from data source " + self.file_path
        )
        if self.datatype == 'PointData':
            temp = vtkimage.GetPointData().GetScalars(self.name)
            self.res = np.array(vtkimage.GetDimensions())
            raw_array = numpy_support.vtk_to_numpy(temp)

            # self.v = normalize(raw_array, full_normalize=True)
            # TODO 先不做数据的归一化
            # self.v = standardization(raw_array)
            self.v = raw_array
            # print(np.min(self.v), np.max(self.v))
            self.v = self.v.reshape(self.res)

        if self.datatype == 'CellData':
            # TODO 
            pass

    def __len__(self):
        return 1

    def __getitem__(self, item):
        # 插值出来的batch_size
        # TODO 随机点三线性插值得到训练数据
        # points = get_random_points_inside_domain(num_points=self.batch_points_num,
        #                                          domain_min=vec3f(-1.), domain_max=vec3f(1.))
        # x, y, z = get_grid_xyz(vec3f(-1.), vec3f(1.), self.res)
        #
        # res = interp3(x, y, z, self.v, points)
        # res = np.expand_dims(res, axis=1)
        # return points, res'
        # 直接取值
        # TODO 根据item输出某个坐标
        point = index_to_domain_xyz(item, vec3f(0.0), vec3f(1.0), self.res)
        xyz = index_to_domain_xyz_index(item, self.res)
        val = self.v[xyz[0], xyz[1], xyz[2]]
        return point, np.array([val])

    def get_volume_data(self):
        return self.v

    def get_volume_res(self):
        return self.res


class EcmwfDataset(Dataset):
    def __init__(self, filename, AttributesName, time_step):
        super(EcmwfDataset, self).__init__()
        self.filename = './data/ecmwf/'+filename
        self.time_step = time_step
        self.Attributes = AttributesName
        self.v = None
        self.res = None
        self.time = 0

    def get_time_stamp(self):
        f = nc.Dataset(self.filename)
        return f.dimensions['time'].size

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return item

    def read_volume_data(self):
        f = nc.Dataset(self.filename)
        longitude = f.dimensions['longitude'].size
        latitude = f.dimensions['latitude'].size
        level = f.dimensions['level'].size
        time = f.dimensions['time'].size
        # 转为np array
        self.res = np.array([longitude, latitude, level])
        self.time = time
        self.v = np.array(f.variables[self.Attributes])[self.time_step, ...]
        self.v = self.v.transpose(2, 1, 0)
        print(np.min(self.v), np.max(self.v))

    def get_volume_res(self):
        return self.res

    def get_volume_data(self):
        return self.v


class CombustionDataset(Dataset):
    def __init__(self, attribute, timestamp):
        super(CombustionDataset, self).__init__()
        self.attribute = attribute
        self.timestamp = timestamp
        self.v = None
        self.res = np.array([480, 720, 120])

    def read_volume_data(self):
        attribute_name = 'jet_'+self.attribute
        path = './data/Combustion'+'/'+attribute_name+'/'
        # read dat 文件
        self.v = read_dat(path+attribute_name+'_{:04d}.dat'.format(self.timestamp))

    def normalize(self):
        # TODO 处理数据
        # self.v = normalize(self.v)
        print(np.min(self.v), np.max(self.v))

    def __getitem__(self, item):
        return item

    def __len__(self):
        return 1

    def get_volume_data(self):
        return self.v

    def get_volume_res(self):
        return self.res


# TODO 多个时间步
class MultiTimeStampDataset:
    # 现在仅支持单一数据集的多时间步
    def __init__(self, DatasetName, AttributesName, batch_points_num=1024*2, datatype='PointData', timestamp=None):
        self.DatasetName = DatasetName
        self.Attributes = AttributesName
        self.datatype = datatype
        self.timestamp = timestamp
        self.batch_points_num = batch_points_num
        self.MultiStamp = []
        #
        if self.DatasetName == 'asteroid':
            files = glob('./data/'+DatasetName+'/*')
            for file in files:
                self.MultiStamp.append(AsteroidDataset(file, AttributesName,
                                                       batch_points_num, datatype, timestamp))
        else:
            print("No this dataset:", self.DatasetName)

    def read_data(self):
        for stamp in self.MultiStamp:
            stamp.read_volume_data()

    def get_volume_data(self):
        data = []
        for stamp in self.MultiStamp:
            data.append(stamp.v)
        return np.array(data)

    def get_volume_res(self):
        time = len(self.MultiStamp)
        res = self.MultiStamp[0].get_volume_res()
        return np.append(time, res)


# # TODO 分解这个数据集,降低显存的使用
# class MetaDataset(Dataset):
#     def __init__(self, data_block_array, maml_chunk=1024*8):
#         super(MetaDataset, self).__init__()
#         self.data_block_array = data_block_array
#         self.chunk = maml_chunk
#         vol = []
#         num = len(self.data_block_array)
#         for data_block in self.data_block_array:
#             vol.append(data_block.v.flatten())
#         vol = np.expand_dims(np.array(vol), axis=-1)
#         pos = get_query_coords(vec3f(-1), vec3f(1), self.data_block_array[0].res).reshape([-1, 3])
#         pos = np.expand_dims(pos, 0).repeat(num, axis=0)
#         # TODO 分 batch_size
#         self.res = np.concatenate((pos, vol), axis=-1)

#     def __len__(self):
#         num_points = self.data_block_array[0].res.prod()
#         return int(np.ceil(num_points/self.chunk))

#     def __getitem__(self, item):
#         # block层级之间 shuffle
#         # np.random.shuffle(self.res[:, ...])
#         return self.res[:, item*self.chunk:(item+1)*self.chunk, ...]
    
class MetaDataset(Dataset):
    def __init__(self, data_block_array, maml_chunk=1024*8):
        super(MetaDataset, self).__init__()
        self.data_block_array = data_block_array
        self.chunk = maml_chunk
        self.max = 0
        vol = []
        num = len(self.data_block_array)
        for data_block in self.data_block_array:
            vol.append(data_block.v.flatten())
            if self.max < data_block.v.max():
                self.max = data_block.v.max()
        vol = np.expand_dims(np.array(vol), axis=-1)
        pos = get_query_coords(vec3f(-1), vec3f(1), self.data_block_array[0].res).reshape([-1, 3])
        pos = np.expand_dims(pos, 0).repeat(num, axis=0)
        # TODO 分 batch_size
        self.res = np.concatenate((pos, vol), axis=-1)

    def __len__(self):
        return len(self.data_block_array)

    def __getitem__(self, item):
        # block层级之间 shuffle
        # np.random.shuffle(self.res[:, ...])
        return self.res[item]
    
    def getmax(self):
        return self.max
    
class ArBubbleDataset(Dataset):
    def __init__(self, DatasetName) -> None:
        super().__init__()
        self.DatasetName = DatasetName
        self.scalar_data = []
        self.file_path = f'./data/'+DatasetName

    def read_data(self):
        files_path = glob(self.file_path+'/*.dat')
        thresold = 1
        i=0
        for path in files_path:
            i += 1
            if i > thresold:
                break
            print("using" + " volume data from data source " + path)
            file = open(path,'rb')
            raw_data = file.read()
            data = np.array([struct.unpack('<f', raw_data[i:i+4]) for i in range(0, len(raw_data), 4)])
            normalized_data = (data - data.min()) / (data.max() - data.min())
            # normalized_data = data
            normalized_data = normalized_data.reshape(640,256,256)
            self.scalar_data.append(normalized_data)
            

    def get_scalar_data(self):
        return np.array(self.scalar_data)
    
    def get_data_res(self):
        return np.append(len(self.scalar_data), self.scalar_data[0].shape)

# 单例测试
def main():
    # d = MultiTimeStampDataset('asteroid', 'v02')
    # d.read_data()
    # BlockGenerator(d.get_volume_data(), d.get_volume_res())
    # print(d.get_volume_data().shape)
    # a = CombustionDataset()
    # a.read_volume_data()
    # a.normalize()
    # a = EcmwfDataset('201810_all_variables.nc', 't', 10)
    # a.read_volume_data()
    a = CombustionDataset()
    a.read_volume_data()
    a.normalize()


if __name__ == '__main__':
    main()

