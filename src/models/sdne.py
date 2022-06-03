# -*- coding: utf-8 -*-
"""

Reference:

    [1] Wang D, Cui P, Zhu W. Structural deep network embedding[C]//Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016: 1225-1234.(https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)


"""
import torch
from .basemodel import GraphBaseModel
from ..utils import process_nxgraph
import numpy as np
import scipy.sparse as sparse
from ..utils import Regularization


class SDNEModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, alpha, beta, device="cpu"):

        super(SDNEModel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        input_dim_copy = input_dim
        layers = []
        for layer_dim in hidden_layers: #[512,64]
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for layer_dim in reversed(hidden_layers[:-1]):
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim

        layers.append(torch.nn.Linear(input_dim, input_dim_copy))
        layers.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*layers)


    def forward(self, A, L):

        Y = self.encoder(A)
        A_hat = self.decoder(Y)

        beta_matrix = torch.ones_like(A)
        mask = A != 0
        beta_matrix[mask] = self.beta
        loss_2nd = torch.mean(torch.sum(torch.pow((A - A_hat) * beta_matrix, 2), dim=1))
        loss_1st =  self.alpha * 2 * torch.trace(torch.matmul(torch.matmul(Y.transpose(0,1), L), Y))
        return loss_2nd + loss_1st




class SDNE(GraphBaseModel):

    def __init__(self, graph, hidden_layers=None, alpha=1e-3, beta=8, gamma=1e-5, device="cuda:0"):
        super().__init__()
        self.graph = graph
        self.idx2node, self.node2idx = process_nxgraph(graph)
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.sdne = SDNEModel(self.node_size, hidden_layers, alpha, beta)
        self.device = device
        self.embeddings = {}
        self.gamma = gamma

        adjacency_matrix, laplace_matrix = self.__create_adjacency_laplace_matrix()
        self.adjacency_matrix = torch.from_numpy(adjacency_matrix.toarray()).float().to(self.device)
        self.laplace_matrix = torch.from_numpy(laplace_matrix.toarray()).float().to(self.device)

    def fit(self, batch_size=512, epochs=100, initial_epoch=0, verbose=1):
        num_samples = self.node_size
        self.sdne.to(self.device)
        optimizer = torch.optim.Adam(self.sdne.parameters())
        if self.gamma:
            regularization = Regularization(self.sdne, gamma=self.gamma)
        if batch_size >= self.node_size:
            batch_size = self.node_size
            print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                batch_size, self.node_size))
            for epoch in range(initial_epoch, epochs):
                loss_epoch = 0
                optimizer.zero_grad()
                loss = self.sdne(self.adjacency_matrix, self.laplace_matrix)
                if self.gamma:
                    reg_loss = regularization(self.sdne)
                    loss = loss + reg_loss
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()
                if (verbose > 0 and epoch%5 == 0):
                    print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4), epoch+1, epochs))
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            for epoch in range(initial_epoch, epochs):
                loss_epoch = 0
                for i in range(steps_per_epoch):
                    idx = np.arange(i * batch_size, min((i+1) * batch_size, self.node_size))
                    A_train = self.adjacency_matrix[idx, :]
                    L_train = self.laplace_matrix[idx][:,idx]
                    # print(A_train.shape, L_train.shape)
                    optimizer.zero_grad()
                    loss = self.sdne(A_train, L_train)
                    loss_epoch += loss.item()
                    loss.backward()
                    optimizer.step()

                if verbose > 0:
                    print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4),
                                                                         epoch + 1, epochs))

    def get_embeddings(self):
        if not self.embeddings:
            self.__get_embeddings()
        embeddings = self.embeddings
        return embeddings

    def __get_embeddings(self):
        embeddings = {}
        with torch.no_grad():
            self.sdne.eval()
            embed = self.sdne.encoder(self.adjacency_matrix)
            for i, embedding in enumerate(embed.cuda().data.cpu().numpy()):
                embeddings[self.idx2node[i]] = embedding
        self.embeddings = embeddings


    def __create_adjacency_laplace_matrix(self):
        node_size = self.node_size
        node2idx = self.node2idx
        adjacency_matrix_data = []
        adjacency_matrix_row_index = []
        adjacency_matrix_col_index = []
        for edge in self.graph.edges():
            v1, v2 = edge
            edge_weight = self.graph[v1][v2].get("weight", 1.0)
            adjacency_matrix_data.append(edge_weight)
            adjacency_matrix_row_index.append(node2idx[v1])
            adjacency_matrix_col_index.append(node2idx[v2])
            
        adjacency_matrix = sparse.csr_matrix((adjacency_matrix_data,
                                              (adjacency_matrix_row_index, adjacency_matrix_col_index)),
                                             shape=(node_size, node_size))

        adjacency_matrix_ = sparse.csr_matrix((adjacency_matrix_data+adjacency_matrix_data,
                                               (adjacency_matrix_row_index+adjacency_matrix_col_index,
                                                adjacency_matrix_col_index+adjacency_matrix_row_index)),
                                              shape=(node_size, node_size))
        degree_matrix = sparse.diags(adjacency_matrix_.sum(axis=1).flatten().tolist()[0])
        laplace_matrix = degree_matrix - adjacency_matrix_
        return adjacency_matrix, laplace_matrix


