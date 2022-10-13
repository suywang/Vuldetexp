import json
import os
import glob

if __name__ == '__main__':
    proportion= 0.5
    src_path='/home/mytest/GNNExplainer-master/gnn_results/'
    dict_path='/home/mytest/nvd/nvd_lindedict/'
    lineinfo='/home/mytest/nvd/nvd_vul_lineinfo.json'
    with open(lineinfo,'r') as lp:
        vul_line=json.load(lp)
    res_files=glob.glob(src_path+'/*.json')
    all_num=0
    true_num=0
    for res_file in res_files:
        flag=0
        with open(res_file,'r') as rp:
            node_list=json.load(rp)
        test_num = int(proportion * len(node_list))+1
        name=res_file.split('/')[-1]
        print(name)
        name2='CVE'+name.split('CVE')[-1].split('@')[0]+'.c'
        name3=name.split('.json')[0]+'.dot.json'
        with open(dict_path+name3,'r') as np:
            node2line=json.load(np)
        gt=vul_line[name2]
        for node in node_list[:test_num]:
            if flag == 1:
                break  
            tmp=0
            for dict_node in node2line:
                if tmp == 2:
                    break
                if dict_node['node_id'] == node:
                    tmp+=1
                    if int(dict_node['line_num']) in gt:
                        true_num+=1
                        flag=1
                        break
        all_num+=1
    print(all_num)
    print(true_num)
    score = true_num/all_num
    print(score)
                    
                


