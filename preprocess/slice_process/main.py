import os
from re import I
from typing import Container
from preprocess import *
from complete_pdg import *
from slice_op import *
from json_to_dot import *
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/home/VulGnnExp/slices_dot/')
    args = parser.parse_args()
    dataset_path = args.input_dir
    json_file = '/home/Dataset/BigVul_new/4_json/1_new/'
    pdg_dot_file = '/home/Dataset/BigVul_new/3_pdg/1_new/'
    
    cpg_dot_path = '/home/Dataset/BigVul_new/3_pdg/1_compdg/'
    sub_graph_path = '/home/Dataset/BigVul_new/5_slices/1_new/'
    ground_truth_path = '/home/Dataset/BigVul_new/5_slices/ground_truth2/'
    #label_pkl = 'nvd_vul_lineinfo.json'
    #label_path = dataset_path + label_pkl
    label_path = '/home/Dataset/BigVul_new/big_vul_new_gt.json'
    dict_path='/home/Dataset/BigVul_new/6_line2node/func_dict/'
    #所有数据
    #container = joern_process(dataset_path+json_file)
    # if os.path.exists('/home/VulGnnExp/Src_code/container.json'):
    #     with open('/home/VulGnnExp/Src_code/container.json','r') as rf:
    #         container = json.load(rf)
    # else:
    #     container = joern_process(json_file)
    #     with open('/home/VulGnnExp/Src_code/container.json','w') as wf:
    #         json.dump(container,wf)
    container = joern_process(json_file)
    i = 0
    sub_cnt = 0
    for data in tqdm(container):
        i += 1
        if data == []:
            sub_cnt += 1
            continue
        data = data[0]
        data_nodes = {}
        idx = data[0]
        cpg = data[1]
        print("===========>>>>>  " + str(i))
        print(idx)


        pdg_edge_list = ddg_edge_genearate(pdg_dot_file, idx)
        if pdg_edge_list == []:
            continue
        data_nodes_tmp = parse_to_nodes(cpg)
        data_nodes = complete_pdg(data_nodes_tmp, pdg_edge_list)

        # generate_complete_json(data_nodes, cpg_dot_path, idx)

        gt_node_list = get_gt_node(data_nodes,dict_path, idx,label_path)
        if gt_node_list != []:
            _gt_slice_list = sup_slice(data_nodes, gt_node_list)
            points_name = '@groundtruth'
            generate_sub_json(data_nodes, _gt_slice_list, ground_truth_path, idx, points_name, label_path)


        # pointer_node_list = get_pointers_node(data_nodes)
        # if pointer_node_list != []:
        #     _pointer_slice_list = pointer_slice(data_nodes, pointer_node_list)
        #     points_name = '@pointer'
        #     generate_sub_json(data_nodes, _pointer_slice_list, sub_graph_path, idx, points_name, label_path)

        # arr_node_list = get_all_array(data_nodes)
        # if arr_node_list != []:
        #     _arr_slice_list = array_slice(data_nodes, arr_node_list)
        #     points_name = '@array'
        #     generate_sub_json(data_nodes, _arr_slice_list, sub_graph_path, idx, points_name, label_path)

        # integer_node_list = get_all_integeroverflow_point(data_nodes)
        # if integer_node_list != []:
        #     _integer_slice_list = inte_slice(data_nodes, integer_node_list)
        #     points_name = '@integer'
        #     generate_sub_json(data_nodes, _integer_slice_list, sub_graph_path, idx, points_name, label_path)

        # call_node_list = get_all_sensitiveAPI(data_nodes)
        # if call_node_list != []:
        #     _call_slice_list = call_slice(data_nodes, call_node_list)
        #     points_name = '@API'
        #     generate_sub_json(data_nodes, _call_slice_list, sub_graph_path, idx, points_name, label_path)            

if __name__ == '__main__':
    main()
