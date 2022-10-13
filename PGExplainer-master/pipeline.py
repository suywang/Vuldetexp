import os
import glob
import time
import torch
from tqdm import tqdm
#from models import GnnNets, GnnNets_NC
from models.Devign import DevignModel, config_model
from utils import PlotUtils
from pgexplainer import PGExplainer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from metrics import top_k_fidelity, top_k_sparsity
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, model_args, train_args


def pipeline_GC(top_k):
    dataset = get_dataset(data_args)
    if data_args.dataset_name == 'mutag':
        data_indices = list(range(len(dataset)))
        pgexplainer_trainset = dataset
    else:
        loader = get_dataloader(dataset, data_args, train_args)
        data_indices = loader['test'].dataset.indices
        pgexplainer_trainset = loader['train'].dataset

    Devign = DevignModel(model_args, max_edge_types=model_args.max_edge_types)
    config_model(Devign, model_args)

    save_dir = os.path.join('/home/mytest/PGExplainer-master/results/', f"nvd_"#f"{data_args.dataset_name}_"
                                         f"{model_args.model_name}_"
                                         f"pgexplainer")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    pgexplainer = PGExplainer(Devign)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()

    pgexplainer.get_explanation_network(pgexplainer_trainset)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    toc = time.perf_counter()
    training_duration = toc - tic
    print(f"training time is {training_duration: .4}s ")

    explain_duration = 0.0
    plotutils = PlotUtils(dataset_name=data_args.dataset_name)
    fidelity_score_list = []
    sparsity_score_list = []
    for data_idx in tqdm(data_indices):
        torch.cuda.empty_cache()
        data = dataset[data_idx]
        #print(data['name'])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tic = time.perf_counter()

        prob = pgexplainer.eval_probs(data.x, data.edge_index, data.edge_attr)
        pred_label = prob.argmax(-1).item()

        if glob.glob(os.path.join(save_dir, f"example_{data_idx}.pt")):
            file = glob.glob(os.path.join(save_dir, f"example_{data_idx}.pt"))[0]
            edge_mask = torch.from_numpy(torch.load(file))
        else:
            edge_mask = pgexplainer.explain_edge_mask(data.x, data.edge_index, data.edge_attr)
            save_path = os.path.join(save_dir, f"example_{data_idx}.pt")
            edge_mask = edge_mask.cpu()
            torch.save(edge_mask.detach().numpy(), save_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        toc = time.perf_counter()
        explain_duration += (toc - tic)

        graph = to_networkx(data)

        fidelity_score = top_k_fidelity(data, edge_mask, top_k, Devign, pred_label)
        sparsity_score = top_k_sparsity(data, edge_mask, top_k)

        fidelity_score_list.append(fidelity_score)
        sparsity_score_list.append(sparsity_score)

        # visualization
        #if hasattr(dataset, 'supplement'):
        #    words = dataset.supplement['sentence_tokens'][str(data_idx)]
        #    plotutils.plot_soft_edge_mask(graph, edge_mask, top_k,
        #                                  x=data.x,
        #                                  words=words,
        #                                  un_directed=True,
        #                                  figname=os.path.join(save_dir, f"example_{data_idx}.png"))
        #else:
        #    plotutils.plot_soft_edge_mask(graph, edge_mask, top_k,
        #                                  x=data.x,
        #                                  edge_index=data.edge_index,
        #                                  un_directed=True,
        #                                  figname=os.path.join(save_dir, f"example_{data_idx}.png"))

    fidelity_scores = torch.tensor(fidelity_score_list)
    sparsity_scores = torch.tensor(sparsity_score_list)
    return fidelity_scores, sparsity_scores


def pipeline_NC(top_k):
    
    #获取数据
    dataset = get_dataset(data_args)
    
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    data = dataset[0]
    node_indices = torch.where(data.test_mask * data.y != 0)[0].tolist()
    
    #获取gnn网络
    gnnNets = GnnNets_NC(input_dim, output_dim, model_args)

    #checkpoint是一个文本文件，记录了训练过程中在所有中间节点上保存的模型的名称，首行记录的是最后（最近）一次保存的模型名称。
    #checkpoint是检查点文件，文件保存了一个目录下所有的模型文件列表
    checkpoint = torch.load(model_args.model_path)
    gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.to_device()
    gnnNets.eval()

    #保存结果
    save_dir = os.path.join('./results', f"{data_args.dataset_name}_"
                                         f"{model_args.model_name}_"
                                         f"pgexplainer")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    #创建解释器
    pgexplainer = PGExplainer(gnnNets)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()

    #训练解释的网络
    pgexplainer.get_explanation_network(dataset, is_graph_classification=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    toc = time.perf_counter()
    training_duration = toc - tic
    print(f"training time is {training_duration}s ")

    #评分
    duration = 0.0
    data = dataset[0]
    fidelity_score_list = []
    sparsity_score_list = []
    plotutils = PlotUtils(dataset_name=data_args.dataset_name)
    for ori_node_idx in tqdm(node_indices):
        tic = time.perf_counter()
        if glob.glob(os.path.join(save_dir, f"node_{ori_node_idx}.pt")):#返回所有匹配的文件路径列表
            file = glob.glob(os.path.join(save_dir, f"node_{ori_node_idx}.pt"))[0]
            edge_mask, x, edge_index, y, subset = torch.load(file)#加载文件到cpu
            edge_mask = torch.from_numpy(edge_mask)#把数组转换成张量,且二者共享内存,对张量进行修改比如重新赋值,那么原始数组也会相应发生改变
            node_idx = int(torch.where(subset == ori_node_idx)[0])#按照一定的规则合并两个tensor类型
            pred_label = pgexplainer.get_node_prediction(node_idx, x, edge_index)
        else:
            x, edge_index, y, subset, kwargs = \
                pgexplainer.get_subgraph(node_idx=ori_node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
            node_idx = int(torch.where(subset == ori_node_idx)[0])

            edge_mask = pgexplainer.explain_edge_mask(x, edge_index)
            pred_label = pgexplainer.get_node_prediction(node_idx, x, edge_index)
            save_path = os.path.join(save_dir, f"node_{ori_node_idx}.pt")
            edge_mask = edge_mask.cpu()
            cache_list = [edge_mask.numpy(), x, edge_index, y, subset]
            torch.save(cache_list, save_path)

        duration += time.perf_counter() - tic
        sub_data = Data(x=x, edge_index=edge_index, y=y)

        graph = to_networkx(sub_data)

        fidelity_score = top_k_fidelity(sub_data, edge_mask, top_k, gnnNets, pred_label,
                                        node_idx=node_idx, undirected=True)
        #sparsity_score = top_k_sparsity(sub_data, edge_mask, top_k, undirected=True)

        #fidelity_score_list.append(fidelity_score)
        #sparsity_score_list.append(sparsity_score)

        # visualization
        #plotutils.plot_soft_edge_mask(graph, edge_mask, top_k,
        #                              y=sub_data.y,
        #                              node_idx=node_idx,
        #                              un_directed=True,
        #                              figname=os.path.join(save_dir, f"example_{ori_node_idx}.png"))
    #貌似是解释研究的分数
    #fidelity_scores = torch.tensor(fidelity_score_list)#生成新的张量
    #sparsity_scores = torch.tensor(sparsity_score_list)
    #return fidelity_scores, sparsity_scores


def pipeline(top_k):
    if data_args.dataset_name.lower() == 'BA_shapes'.lower():
        rets = pipeline_NC(top_k)
    else:
        rets = pipeline_GC(top_k)
    return rets


if __name__ == '__main__':
    top_k = 6
    fidelity_scores, sparsity_scores = pipeline(top_k)
    #解释的评价分数
    #print(f"fidelity score: {fidelity_scores.mean().item():.4f}, "
    #      f"sparsity score: {sparsity_scores.mean().item():.4f}")
