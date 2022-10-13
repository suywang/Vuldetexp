import glob 
import os,sys
from torch_geometric.data import Data, InMemoryDataset
import json
import torch

def read_json(filename):
    #读取文件
    with open(filename,'r') as f:
        file = json.load(f)
    #文件内容读取到torch.tensor()中
    x = torch.tensor(file['node_features'],dtype=torch.float64)

    edge_index_list = []
    for edge in file['graph']:
        edge_index_list.append([edge[0],edge[2]])
    edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()
    
    edge_attr_list = []
    for edge in file['graph']:
        edge_attr_list.append([edge[1]])
    edge_attr = torch.tensor(edge_attr_list)
    
    y=[]
    y.append([file['target']])
    y=torch.tensor(y)
    
    data=Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y)
    #torch.save(data,filename+'.pt')
    return data

def main():
    data_list = []
        
    # json -> data_list
    dataset_path=['/home/devign_out/devign_out_qu_vul/','/home/devign_out/devign_out_qu_novul/',
                   '/home/devign_out/devign_out_ff_vul/','/home/devign_out/devign_out_ff_novul/']
    for path in dataset_path:
        dataset_list = glob.glob(path + '*.json')
        i = 0
        for data_name in dataset_list:
            i += 1
            if i>=10:
                 break
            data_list.append(read_json(data_name))

if __name__ == '__main__':
    main()