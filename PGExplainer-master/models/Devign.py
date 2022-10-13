import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GatedGraphConv
from dgl import DGLGraph


def config_model(model, args):
    model.to(args.device_id)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt)
    print(f'Loading best checkpoint ... ')


class DevignModel(nn.Module):
    def __init__(self, model_args, input_dim=100, output_dim=200, max_edge_types=4, num_steps=8):
        super(DevignModel, self).__init__()
        self.device = model_args.device
        self.inp_dim = input_dim
        self.out_dim = output_dim
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

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=2)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def de_batchify_graphs(self, features=None):
        if features is None:
            features = self.graph.ndata['features']
        assert isinstance(features, torch.Tensor)

        vectors = [torch.tensor(1)]
        vectors[0] = torch.tensor(features,requires_grad = True)
        output_vectors = torch.stack(vectors)

        return output_vectors

    def get_network_inputs(self, graph, cuda=False, device=None):
        features = graph.ndata['features']
        edge_types = graph.edata['etype']
        if cuda:
            self.cuda(device=device)
            return graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return graph, features, edge_types
        pass

    def forward(self, batch, cuda=False):
        #graph, features, edge_types, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        graph, features, edge_types = self.get_network_inputs(batch, cuda=cuda)
        graph = graph.to(torch.device("cuda:0"))
        features = features.to(torch.device("cuda:0"))
        edge_types = edge_types.to(torch.device("cuda:0"))
        outputs = self.ggnn(graph, features, edge_types)
        x_i = self.de_batchify_graphs(features)
        h_i = self.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            F.relu(
                self.batchnorm_1d(
                    self.conv_l1(h_i.transpose(1, 2))  # num_node >= 5
                )
            )
        )
        Y_2 = self.maxpool2(
            F.relu(
                self.batchnorm_1d(
                    self.conv_l2(Y_1)
                )
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            F.relu(
                self.batchnorm_1d_for_concat(
                    self.conv_l1_for_concat(c_i.transpose(1, 2))
                )
            )
        )
        Z_2 = self.maxpool2_for_concat(
            F.relu(
                self.batchnorm_1d_for_concat(
                    self.conv_l2_for_concat(Z_1)
                )
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg)
        return avg, result, features