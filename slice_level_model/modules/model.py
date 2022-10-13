import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types=2, num_steps=6):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim #100
        self.out_dim = output_dim #200
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.batchnorm_1d = torch.nn.BatchNorm1d(output_dim)
        self.batchnorm_1d_for_concat = torch.nn.BatchNorm1d(self.concat_dim)
        
        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        graph = graph.to(torch.device("cuda:0"))
        features = features.to(torch.device("cuda:0"))
        edge_types = edge_types.to(torch.device("cuda:0"))
        outputs = self.ggnn(graph, features, edge_types)
        x_i, _ = batch.de_batchify_graphs(features)
        h_i, _ = batch.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.batchnorm_1d(
                    self.conv_l1(h_i.transpose(1, 2)) #outputs
                )
                
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.batchnorm_1d(
                    self.conv_l2(Y_1) #outputs
                )
                
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.batchnorm_1d_for_concat(
                    self.conv_l1_for_concat(c_i.transpose(1, 2)) #ouputs+feature
                )
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.batchnorm_1d_for_concat(   
                    self.conv_l2_for_concat(Z_1) #ouputs+feature
                )
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result


class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(torch.device("cuda:0"))
        features = features.to(torch.device("cuda:0"))
        edge_types = edge_types.to(torch.device("cuda:0"))
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result
