import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
import utils
import math
import threading

from scipy.stats import entropy
from tqdm import tqdm
from collections import Counter
from scipy.spatial.distance import jensenshannon

class Variable_MI():
    def __init__(self) -> None:
        return
    
    def variable_entropy(self, block) -> float:
        flattened_block = block.flatten()
        counts = Counter(flattened_block)
        probabilities = np.array(list(counts.values())) / len(flattened_block)
        # hist, _ = np.histogram((flattened_block*256).astype(int), bins=range(257))
        # prob = hist/hist.sum()
        return entropy(probabilities)

    def two_variable_MI(self, block1, block2) -> float:
        a = block1.reshape(-1)
        b = block2.reshape(-1)
        hist, x_edges, y_edges = np.histogram2d(a,b,bins=[50,50])
        return entropy(hist.flatten())
    
    def HSIC(self, K, L) -> float:
        point_num = K.shape[0]
        H = np.eye(point_num) - np.dot(np.ones(point_num), np.ones(point_num).T)*(1/point_num)
        mean_K = np.dot(np.dot(H,K),H)
        mean_L = np.dot(np.dot(H,L),H)
        return np.dot(mean_K.flatten(), mean_L.flatten())/((point_num-1) ** 2)
    
    def CKA(self, matrix1, matrix2) -> float:
        K = np.dot(matrix1, matrix1.T)
        L = np.dot(matrix2, matrix2.T)
        return self.HSIC(K,L)/((self.HSIC(K,K)*self.HSIC(L,L)) ** (1/2))
    
    def jsd(self, block1, block2):
        a = block1.reshape(-1)
        b = block2.reshape(-1)
        dist_a, _ = np.histogram(a, bins=1000)
        dist_b, _ = np.histogram(b, bins=1000)
        return jensenshannon(dist_a, dist_b)
        
    def variables_MI_heat_map(self, block_array):
        block_num = block_array.shape[0]
        heat_map = np.empty((block_num,block_num))
        for i in tqdm(range(block_num)):
            for j in range(block_num):
                heat_map[i][j] = self.two_variable_MI(block_array[i],block_array[j])

        # draw heat map
        plt.clf()
        plt.imshow(heat_map, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        def close_figure():
            plt.close()  # 关闭 matplotlib 窗口
        fig = plt.gcf()
        timer = fig.canvas.new_timer(interval=5000)
        timer.add_callback(close_figure)
        timer.start()
        plt.show() 
        return heat_map
    
    def variables_CKA_heat_map(self, block_array):
        block_num = block_array.shape[0]
        heat_map = np.empty((block_num,block_num))
        for i in tqdm(range(block_num)):
            for j in range(block_num):
                heat_map[i][j] = self.CKA(block_array[i],block_array[j])

        # draw heat map
        plt.clf()
        plt.imshow(heat_map, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        def close_figure():
            plt.close()  # 关闭 matplotlib 窗口
        fig = plt.gcf()
        timer = fig.canvas.new_timer(interval=5000)
        timer.add_callback(close_figure)
        timer.start()
        plt.show() 
        return heat_map
    
class Network_Latent_Vector():
    def __init__(self, templates, layer_index, args) -> None:
        self.templates = templates
        self.layer_index = layer_index
        self.args = args
        self.templates_latent_vector = []
        self.hook_handles = []

    def sample(self, point_num):
        cube_res = [math.ceil(point_num ** (1/3)) for i in range(3)]
        return utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), cube_res).reshape([-1, 3])
        

    def hook_fn(self, module, input, output):
        latent_vector = output
        self.templates_latent_vector.append(latent_vector)
        return
    
    def register_hook(self):
        for template in self.templates:
            intermediate_layer = template[0].siren.pts_linear[self.layer_index]
            self.hook_handles.append(intermediate_layer.register_forward_hook(self.hook_fn))

    def get_latent_vectors(self):
        for template in self.templates:
            network_query_fn = template[1]
            x = torch.from_numpy(self.sample(64)).to(torch.float32).to(utils.get_device(self.args.GPU))
            network_query_fn(x, template[0])

        return self.templates_latent_vector


    def remove_hook(self):
        for handle in self.hook_handles:
            handle.remove()
    
