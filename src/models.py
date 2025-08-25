import dgl
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCNConv, GINConv
from layers import AttentionModule, TensorNetworkModule
from math import exp
import math
from GedMatrix import GedMatrixModule

class Grasp(torch.nn.Module):
    def __init__(self,args,number_of_labels):
        super(Grasp,self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
    
    def init_mlp_features(self):
        k = self.number_labels+self.args.filters_3+self.args.filters_3+self.args.filters_3+self.args.filters_3+self.args.filters_3+self.args.filters_3
        layers = []
        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k,self.args.filters_3))
        self.mlp = torch.nn.Sequential(*layers)
    
    def setup_layers(self):
        self.args.gnn_operator = 'gin'
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.args.filters_3, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin': 
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))
            
            nn4 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))
            
            nn5 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
            self.convolution_4 = GINConv(nn4, train_eps=True)
            self.convolution_5 = GINConv(nn5, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')
        self.init_mlp_features()
        
        self.bilinear_dim = 32
        self.pattern_att_dim = 32
        self.cost_att_dim = 32
        
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)    
        
        self.h0_pattern_Matrix = GedMatrixModule(self.args.filters_3, self.bilinear_dim)
        self.h1_pattern_Matrix = GedMatrixModule(self.args.filters_3, self.bilinear_dim)
        self.h2_pattern_Matrix = GedMatrixModule(self.args.filters_3, self.bilinear_dim)
        self.h3_pattern_Matrix = GedMatrixModule(self.args.filters_3, self.bilinear_dim)
        self.h4_pattern_Matrix = GedMatrixModule(self.args.filters_3, self.bilinear_dim)
        self.h5_pattern_Matrix = GedMatrixModule(self.args.filters_3, self.bilinear_dim)
        
        self.gru = torch.nn.GRU(self.bilinear_dim, self.bilinear_dim, bidirectional=False, batch_first=True)
        
        self.cost_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.cost_att_dim, self.cost_att_dim * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.cost_att_dim * 2, self.cost_att_dim ),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.cost_att_dim, 1)
                )
        
        self.map_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.bilinear_dim, self.bilinear_dim * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.bilinear_dim * 2, self.bilinear_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.bilinear_dim, 1)
                )
        
        
        self.W_attn = torch.nn.Linear(self.bilinear_dim, self.pattern_att_dim, bias=True)
        self.v_attn = torch.nn.Parameter(torch.randn(self.pattern_att_dim))

        self.W_q1 = torch.nn.Linear(self.bilinear_dim, self.cost_att_dim, bias=True)
        self.W_q2 = torch.nn.Linear(self.cost_att_dim, self.cost_att_dim, bias=True)
        self.W_k = torch.nn.Linear(self.bilinear_dim, self.cost_att_dim, bias=True)
        self.W_v = torch.nn.Linear(self.bilinear_dim, self.cost_att_dim, bias=True)
        
        self.layer0_emb = torch.nn.Sequential(
            torch.nn.Linear(self.number_labels + 1, self.args.filters_3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.filters_3, self.args.filters_3)
        )
    
    def convolutional_pass(self, edge_index, features):
        if self.args.dataset == 'IMDB':
            using_dropout = False
        else:
            using_dropout = self.training
            
        features_list = features
        
        degree_mat = torch.zeros(features.size()[0], features.size()[0])
        for i in range(edge_index.size()[1]-features.size()[0]):
            degree_mat[edge_index[0][i]][edge_index[1][i]] = 1
        degree_mat = torch.sum(degree_mat, dim=1).unsqueeze(1)
        degree = torch.cat((features, degree_mat), dim=1)
        features0 = self.layer0_emb(degree)
        features_list = torch.cat([features_list,features0],dim=1)
        
        features1 = self.convolution_1(features0, edge_index)
        features_list = torch.cat([features_list,features1],dim=1)
        features11 = torch.nn.functional.relu(features1)
        features11 = torch.nn.functional.dropout(features11,
                                               p=self.args.dropout,
                                               training=using_dropout)

        features2 = self.convolution_2(features11, edge_index)
        features_list = torch.cat([features_list,features2],dim=1)
        features22 = torch.nn.functional.relu(features2)
        features22 = torch.nn.functional.dropout(features22,
                                               p=self.args.dropout,
                                               training=using_dropout)

        features3 = self.convolution_3(features22, edge_index)
        features_list = torch.cat([features_list,features3],dim=1)
        features33 = torch.nn.functional.relu(features3)
        features33 = torch.nn.functional.dropout(features33,
                                               p=self.args.dropout,
                                               training=using_dropout)
        
        features4 = self.convolution_4(features33, edge_index)
        features_list = torch.cat([features_list,features4],dim=1)
        features44 = torch.nn.functional.relu(features4)
        features44 = torch.nn.functional.dropout(features44,
                                               p=self.args.dropout,
                                               training=using_dropout)
        
        features5 = self.convolution_5(features44, edge_index)
        features_list = torch.cat([features_list,features5],dim=1)
        
        features = self.mlp(features_list)
        
        return features, features0, features1, features2, features3, features4, features5
        
    def get_bias(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)
       
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score   
    
    def zero_padding(self, g1_h, g2_h):
        max_len = max(len(g1_h), len(g2_h))
        new_tensor1 = torch.zeros(max_len, 32)
        new_tensor2 = torch.zeros(max_len, 32)
        new_tensor1[:len(g1_h), :32] = g1_h
        new_tensor2[:len(g2_h), :32] = g2_h
        return new_tensor1, new_tensor2
    
    def forward(self,data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1, g1_h0, g1_h1, g1_h2, g1_h3, g1_h4, g1_h5 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2, g2_h0, g2_h1, g2_h2, g2_h3, g2_h4, g2_h5 = self.convolutional_pass(edge_index_2, features_2)
        
        h_pattern_list = []
        h_pattern_list.append(self.h0_pattern_Matrix(g1_h0, g2_h0))
        h_pattern_list.append(self.h1_pattern_Matrix(g1_h1, g2_h1))
        h_pattern_list.append(self.h2_pattern_Matrix(g1_h2, g2_h2))
        h_pattern_list.append(self.h3_pattern_Matrix(g1_h3, g2_h3))
        h_pattern_list.append(self.h4_pattern_Matrix(g1_h4, g2_h4))
        h_pattern_list.append(self.h5_pattern_Matrix(g1_h5, g2_h5))
        h_pattern_stack = torch.stack(h_pattern_list, dim=0)
        
        seq_len, n, m, feature_dim = h_pattern_stack.shape
        gru_input = h_pattern_stack.permute(1, 2, 0, 3).view(n * m, seq_len, feature_dim)
        out, hidden = self.gru(gru_input)
        
        score = torch.tanh(self.W_attn(out))
        score = torch.matmul(score, self.v_attn)
        a_t = torch.softmax(score, dim=1).unsqueeze(-1)
        context = (a_t * out).sum(dim=1) 
        pattern = context.view(n, m, self.bilinear_dim)
        
        q = self.W_q2(torch.relu(self.W_q1(pattern)))
        k = self.W_k(h_pattern_stack)
        v = self.W_v(h_pattern_stack)
        s = (q * k).sum(dim=-1) / math.sqrt(self.bilinear_dim)
        a = torch.softmax(s, dim=0)
        context = (a.unsqueeze(-1) * v).sum(dim=0)
        
        cost_matrix = self.cost_mlp(context).squeeze(-1)
        map_matrix = self.map_mlp(pattern).squeeze(-1)
        
        softmax_row = torch.nn.Softmax(dim=1)
        softmax_col = torch.nn.Softmax(dim=0)
        
        row_agg = softmax_row(map_matrix) * cost_matrix
        col_agg = softmax_col(map_matrix) * cost_matrix

        bias_value = self.get_bias(abstract_features_1, abstract_features_2)

        score = torch.sigmoid(((row_agg.sum() + col_agg.sum())/2) + bias_value)
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item(), map_matrix