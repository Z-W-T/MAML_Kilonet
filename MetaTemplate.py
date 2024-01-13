import torch
from torch import nn
from collections import OrderedDict
from networks import create_mlp
import numpy as np
from utils import *
from copy import deepcopy


def clip_grad_by_norm(grad, max_norm):
    """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
    """

    total_norm = 0
    counter = 0
    for g in grad:
        param_norm = g.data.norm(2)
        total_norm += param_norm.item() ** 2
        counter += 1
    total_norm = total_norm ** (1. / 2)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grad:
            g.data.mul_(clip_coef)

    return total_norm / counter


class MetaTemplate(nn.Module):
    """
        Meta Learner
    """
    def __init__(self, module, network_query_fn, loss, args, update_step_test=10, meta_lr=1e-3, update_lr=1e-3, num_meta_steps
    =5):
        super(MetaTemplate, self).__init__()
        self.meta_lr = meta_lr
        self.update_lr = update_lr
        # start, network_query_fn, grad_vars, optimizer, model
        self.net = module
        self.network_query_fn = network_query_fn
        self.num_meta_steps = num_meta_steps
        self.loss = loss
        self.update_step_test = update_step_test
        self.args = args
        self.parameters = [parameter for name, parameter in self.net.named_parameters()]
        # linear+output
        self.forward_parameters = self.parameters[:self.net.D*2+2]
        #input+output
        self.transformer_parameters = self.parameters[self.net.D*2+4:]
        # input transformer + base network + output transformer
        self.MTF_parameters = self.parameters[self.net.D*2+4:]

        self.forward_optimizer = torch.optim.Adam(self.parameters[:self.net.D*2+2], lr=self.meta_lr)
        self.transformer_optimizer = torch.optim.Adam(self.parameters[:self.net.D*2], lr=self.meta_lr)
        self.modulated_optimizer = torch.optim.Adam(self.parameters[:self.net.D*2]+self.parameters[self.net.D*2+2:self.net.D*2+4], lr=self.meta_lr)
        self.MTF_optimizer = torch.optim.Adam(self.parameters[:self.net.D*2]+self.parameters[self.net.D*2+2:self.net.D*2+4], lr=self.meta_lr)
        self.meta_optim = torch.optim.Adam(self.parameters, lr=self.meta_lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optim, step_size=250, gamma=0.7)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        for i in range(task_num):
            # predictions = self.net(x_spt[i])
            # loss = self.loss(predictions, y_spt[i])
            # # 引入二阶导数
            # grad = torch.autograd.grad(loss, self.forward_parameters, create_graph=True)
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.forward_parameters)))
            # fast_weights = self.forward_parameters

            # for k in range(self.num_meta_steps):
            #     predictions = self.net(x_spt[i], fast_weights)
            #     loss = self.loss(predictions, y_spt[i])
            #     # 2. compute grad on theta_pi
            #     grad = torch.autograd.grad(loss, fast_weights)
            #     # 3. theta_pi = theta_pi - train_lr * grad
            #     fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
            # predictions = self.net(x_qry[i], fast_weights)
            predictions = self.net(x_qry[i])
            loss_q = self.loss(predictions, y_qry[i])
            losses_qry += loss_q

        # optimize theta parameters
        losses_qry = losses_qry / task_num
        self.forward_optimizer.zero_grad()
        losses_qry.backward()
        self.forward_optimizer.step()
        return losses_qry.item()
    
    def modulated_forward(self, x_spt, y_spt, x_qry, y_qry, modulations):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        for i in range(task_num):
            modulation = modulations[i]
            for k in range(1, self.num_meta_steps):
                predictions = self.net.modulated_forward(x_spt[i], modulation)
                loss = self.loss(predictions, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, modulation)
                # 3. theta_pi = theta_pi - train_lr * grad
                modulation = modulation - self.update_lr*grad[0]
            predictions = self.net.modulated_forward(x_qry[i], modulation)
            loss_q = self.loss(predictions, y_qry[i])
            losses_qry += loss_q

        # optimize theta parameters
        losses_qry = losses_qry / task_num
        self.modulated_optimizer.zero_grad()
        losses_qry.backward()
        self.modulated_optimizer.step()
        # self.scheduler.step()
        return losses_qry.item()
    
    def transformer_forward(self, x_spt, y_spt, x_qry, y_qry):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        for i in range(task_num):
            fast_weights = self.transformer_parameters
            for k in range(self.num_meta_steps):
                predictions = self.net.transformer_forward(x_spt[i], fast_weights)
                loss = self.loss(predictions, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            predictions = self.net.transformer_forward(x_qry[i], fast_weights)
            loss_q = self.loss(predictions, y_qry[i])
            losses_qry += loss_q

        # optimize theta parameters
        losses_qry = losses_qry / task_num
        self.forward_optimizer.zero_grad()
        losses_qry.backward()
        self.forward_optimizer.step()
        # self.scheduler.step()
        return losses_qry.item()
    
    def MTF_forward(self, x_spt, y_spt, x_qry, y_qry, modulations):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        for i in range(task_num):
            modulation = modulations[i]
            fast_weights = self.transformer_parameters+[modulation]
            for k in range(self.num_meta_steps):
                predictions = self.net.MTF_forward(x_spt[i], fast_weights)
                loss = self.loss(predictions, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[:-1], fast_weights[:-1])))+ [fast_weights[-1] - self.update_lr * 100 * grad[-1]]
            predictions = self.net.MTF_forward(x_qry[i], fast_weights)
            loss_q = self.loss(predictions, y_qry[i])
            losses_qry += loss_q

        # optimize theta parameters
        losses_qry = losses_qry / task_num
        self.MTF_optimizer.zero_grad()
        losses_qry.backward()
        self.MTF_optimizer.step()
        # self.scheduler.step()
        return losses_qry.item()
    
    def variant_forward(self, x_spt, y_spt, x_qry, y_qry, modulations, epoch, pattern):
        if pattern == "F":
            return self.forward(x_spt, y_spt, x_qry, y_qry) 
        elif pattern == "MF":
            if epoch < self.args.maml_boundary:
                return self.modulated_forward(x_spt, y_spt, x_qry, y_qry, modulations)
            else:
                return self.MTF_forward(x_spt, y_spt, x_qry, y_qry, modulations)
        elif pattern == "TF":
            if epoch < self.args.maml_boundary:
                return self.transformer_forward(x_spt, y_spt, x_qry, y_qry)
            else:
                return self.MTF_forward(x_spt, y_spt, x_qry, y_qry, modulations) 
        elif pattern == "MTF":
                return self.MTF_forward(x_spt, y_spt, x_qry, y_qry, modulations)   
    
    def modulation_forward(self, x_spt, y_spt, x_qry, y_qry, modulations):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        for i in range(task_num):
            modulation = modulations[i]
            for k in range(1, self.num_meta_steps):
                predictions = self.net.modulated_forward(x_spt[i], modulation)
                loss = self.loss(predictions, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, modulation)
                # 3. theta_pi = theta_pi - train_lr * grad
                modulation = modulation - self.update_lr*grad[0]
            modulations[i]=modulation
        return modulations
    
    def modulation_transformer_forward(self, x_spt, y_spt, x_qry, y_qry, MTF_parameters=None):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        fast_weights_array = []
        for i in range(task_num):
            if MTF_parameters == None:
                fast_weights = self.transformer_parameters
            else:
                fast_weights = MTF_parameters
            for k in range(self.num_meta_steps):
                predictions = self.net.MTF_forward(x_spt[i], fast_weights)
                loss = self.loss(predictions, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[:], fast_weights[:])))
            fast_weights_array.append(fast_weights)
        return fast_weights_array


    # TODO 完善微调步骤
    def finetune(self, x_spt, y_spt, x_qry, y_qry):
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetune on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = self.network_query_fn(x_spt, net)
        loss = self.loss(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = self.network_query_fn(x_spt, net, fast_weights)
            loss = self.loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        del net
        print(y_qry)
        return loss.item()

    def save_model(self, basedir, filename):
        # TODO save model to binary file
        pass


def main():
    pass


if __name__ == '__main__':
    main()








