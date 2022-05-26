# -*- coding: utf-8 -*-
import torch

def process_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


class Regularization(torch.nn.Module):

    def __init__(self, model, gamma=0.01, p=2, device="cpu"):

        super().__init__()
        if gamma <= 0:
            print("param weight_decay can not be <= 0")
            exit(0)
        self.model = model
        self.gamma = gamma
        self.p = p
        self.device = device
        self.weight_list = self.get_weight_list(model)
        self.weight_info = self.get_weight_info(self.weight_list) 

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, model):
        self.weight_list = self.get_weight_list(model)
        reg_loss = self.regulation_loss(self.weight_list, self.gamma, self.p)
        return reg_loss

    def regulation_loss(self, weight_list, gamma, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss += l2_reg
        reg_loss = reg_loss * gamma
        return reg_loss

    def get_weight_list(self, model):
        weight_list = []
        for name, param in model.named_parameters():

            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def get_weight_info(self, weight_list):
        print("#"*10, "regulations weight", "#"*10)
        for name, param in weight_list:
            print(name)
        print("#"*25)