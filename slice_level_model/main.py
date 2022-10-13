import argparse
import os
import pickle
import sys
import joblib
import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import train, eval
from utils import tally_param, debug

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import torch.nn.functional as F

if __name__ == '__main__':
    torch.manual_seed(22)
    np.random.seed(22)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.',default='devign')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/home/Devign-master_git/dataset')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')
    parser.add_argument('--subpdg_tag', type=str, help='Name of the node feature.', default='subpdg')
    parser.add_argument('--subpdg_num_tag', type=str, help='Name of the node feature.', default='subpdg_num')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=8)
    parser.add_argument('--task', type=str, help='train or pretrain', default='eval')

    args = parser.parse_args()
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = args.input_dir
    if args.task != 'eval':
        processed_data_path = os.path.join('/home/Devign-master_git/data_loader', '522_8_1label_processed_215_test.bin')
        if True and os.path.exists(processed_data_path):
            print('*'*20)
            dataset = joblib.load(open(processed_data_path, 'rb'))
            #debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
            debug('Reading already processed data from %s!' % processed_data_path)
        else:
            print('#'*20)
            dataset = DataSet(train_src=os.path.join(input_dir, 'gnn2.txt'),
                              valid_src=None,
                              test_src=os.path.join(input_dir, 'gnn2_test.txt'),
                              batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                              l_ident=args.label_tag)
            file = open(processed_data_path, 'wb')
            joblib.dump(dataset, file)
            file.close()
    else:
        processed_data_path = os.path.join('/home/Devign-master_git/data_loader', '522_8_1label_processed_215_test.bin')
        if True and os.path.exists(processed_data_path):
            print('*'*20)
            dataset = joblib.load(open(processed_data_path, 'rb'))
            #debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
            debug('Reading already processed data from %s!' % processed_data_path)
        else:
            print('#'*20)
            dataset = DataSet(train_src=None,
                            valid_src=None,
                            test_src=os.path.join(input_dir, 'all_0_522.txt'),
                            batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                            l_ident=args.label_tag)
            file = open(processed_data_path, 'wb')
            joblib.dump(dataset, file)
            file.close()

    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'


    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=3)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    print(model)
    model.cuda()
    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    debug('batch size  : %d' % args.batch_size)
    debug('lr  : 0.0001')
    debug('weight_decay  : 0.0001')
    model_dir = '/home/Devign-master_git/models/'
    if args.task == 'eval':
        eval(model=model, dataset=dataset, max_steps=2000, dev_every=15,
            loss_function=loss_function, optimizer=optim,
            save_path=model_dir , max_patience=100, log_every=None) 
    else:
        train(model=model, dataset=dataset, max_steps=2000, dev_every=15,
              loss_function=loss_function, optimizer=optim,
              save_path=model_dir + '/01-GGNNModel', max_patience=100, log_every=None) 

