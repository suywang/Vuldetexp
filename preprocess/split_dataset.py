import os
import glob
import random
import json
import time
import multiprocessing
import pandas as pd
import re
from multiprocessing import Manager, Pool
import numpy as np
from functools import partial


def split(test_set, val_set, train_set, _file_list, output_path):
    sample_count = len(_file_list)
    print("^^^^^^^^^^^^^^^^^^^^^")
    print(sample_count)

    trian_count = sample_count * 0.8
    val_count = sample_count * 0.1
    test_count = sample_count * 0.1

    while (len(test_set) < test_count):
        x = random.randint(0, sample_count-1)
        if x not in test_set:
            test_set.add(x)
    
    while (len(val_set) < val_count):
        x = random.randint(0, sample_count-1)
        if x in test_set:
            continue
        if x not in val_set:
            val_set.add(x)

    for x in range(sample_count-1):
        if x in test_set or x in val_set:
            continue
        train_set.add(x)

    train_rec = []
    val_rec = []
    test_rec = []
    for i in train_set:
        file_name = _file_list[i]
        train_rec.append(file_name)
    for i in val_set:
        file_name = _file_list[i]
        val_rec.append(file_name)
    for i in test_set:
        file_name = _file_list[i]
        test_rec.append(file_name)
        
    with open(output_path+'test.txt','w') as f_test:
        f_test.writelines(test_rec)
    
    with open(output_path+'val.txt','w') as f_val:
        f_val.writelines(val_rec)
    
    with open(output_path+'train.txt','w') as f_train:
        f_train.writelines(train_rec)



def train_1(train_set, _file_list, train_json):
    ## train
    train_dict = {}
    count_1 = 0
    train_set_count = len(train_set)
    for i in train_set:
        func_dic = {}
        try:
            vul = _file_list[i].rsplit("/")[-1][0]
            path = _file_list[i]+"/"
            _files = glob.glob(path+'*')
            sub_pdg_dict = {}
            func_dic["subpdg_num"] = len(_files)
            func_dic["subpdg"] = sub_pdg_dict
            func_dic["target"] = int(vul)
            
            for file_name in _files:
                sub_pdg_dict[file_name] = {}
                with open(file_name, 'r') as fin:
                    load_dict = json.load(fin)
                sub_pdg_dict[file_name]["node_features"] = load_dict["node_features"]
                sub_pdg_dict[file_name]["graph"] = load_dict["graph"]
            train_dict[path] = func_dic
            count_1 += 1
            if count_1 == 10:
                break
            print("------- LOADING Train Set: " + str(count_1) + '/' + str(train_set_count) + "-------")
        except:
            continue
    print("-> Now your Train set has " + str(len(train_dict)) + " samples!")
    with open(train_json, 'w') as fout:
        json.dump(train_dict, fout)
    print("++++ Train Set Over ++++")


def val_test_1(test_set, val_set, _file_list, test_json, val_json):
    ## test
    test_dict = {}
    count_1 = 0
    test_set_count = len(test_set)
    for i in test_set:
        func_dic = {}
        try:
            vul = _file_list[i].rsplit("/")[-1][0]
            path = _file_list[i]+"/"
            _files = glob.glob(path+'*')
            sub_pdg_dict = {}
            func_dic["subpdg_num"] = len(_files)
            func_dic["subpdg"] = sub_pdg_dict
            func_dic["target"] = int(vul)
            
            for file_name in _files:
                sub_pdg_dict[file_name] = {}
                with open(file_name, 'r') as fin:
                    load_dict = json.load(fin)
                sub_pdg_dict[file_name]["node_features"] = load_dict["node_features"]
                sub_pdg_dict[file_name]["graph"] = load_dict["graph"]
            test_dict[path] = func_dic
            count_1 += 1
            if count_1 == 2:
                break
            print("------- LOADING Test Set: " + str(count_1) + '/ 18460 -------')
        except:
            continue
    print("-> Now your Test set has " + str(len(test_dict)) + " samples!")
    with open(test_json, 'w') as fout:
        json.dump(test_dict, fout)
    print("++++ Train Set Over ++++")

    ## validate
    val_dict = {}
    count_1 = 0
    val_set_count = len(val_set)
    for i in val_set:
        func_dic = {}
        try:
            vul = _file_list[i].rsplit("/")[-1][0]
            path = _file_list[i]+"/"
            _files = glob.glob(path+'*')
            sub_pdg_dict = {}
            func_dic["subpdg_num"] = len(_files)
            func_dic["subpdg"] = sub_pdg_dict
            func_dic["target"] = int(vul)
            
            for file_name in _files:
                sub_pdg_dict[file_name] = {}
                with open(file_name, 'r') as fin:
                    load_dict = json.load(fin)
                sub_pdg_dict[file_name]["node_features"] = load_dict["node_features"]
                sub_pdg_dict[file_name]["graph"] = load_dict["graph"]
            val_dict[path] = func_dic
            count_1 += 1
            if count_1 == 2:
                break
            print("------- LOADING Val Set: " + str(count_1) + '/' + str(val_set_count) + "-------")
        except:
            continue
    print("-> Now your Val set has " + str(len(val_dict)) + " samples!")
    with open(val_json, 'w') as fout:
        json.dump(val_dict, fout)
    print("++++ Train Set Over ++++")

def f(row):
    return sum(row)+a
 
def apply_f(df):
    return df.apply(f,axis=1)

def init_process(global_vars):
    global a
    a = global_vars


def train_2(train_json, file_list):
    ## train
    train_dict = {}
    count_1 = 0
    print(count_1)
    json_data = pd.DataFrame(columns=["node_features", "graph", "target"])
    for index, _file in enumerate(file_list):
        if index < 45000:
            continue
        file_name = _file.replace("\n","")
        if file_name in json_data.index.tolist():
            continue
        try:
            with open (file_name) as subgraph:
                json_text = json.load(subgraph)
                node_features = json_text["node_features"]
                graph = json_text["graph"]
                target = json_text["target"]
            json_data.loc[file_name] = [node_features, graph, target]

            print("------- LOADING Train Set: " + str(index) + '/147671 -------')
            count_1 += 1

            if count_1 % 15000 == 0:
                json_data = json_data.replace(to_replace= r'\\', value= '', regex=True)
                result = json_data.to_json(train_json,orient='index')
                print('!!!~~~~ APPEND SUCCESS !~~~~!!!!')
                break              
        except:
            continue

    json_data = json_data.replace(to_replace= r'\\', value= '', regex=True)
    result = json_data.to_json(orient='index').replace("\\","")
    
    with open(train_json, 'w') as f:
        json.dump(result, f, ensure_ascii=False)  
    print("++++ Train Set Over ++++")         


def train_2_1(file_list,train_json):
    ## train
    train_dict = {}
    count_1 = 0
    for _file in file_list:
        try:
            file_name = _file.replace("\n","")
            with open(file_name, 'r') as fin:
                load_dict = json.load(fin)
            train_dict[file_name] = load_dict
            count_1 += 1
            if count_1 % 15000 == 0:
                with open(train_json, 'w') as fout:
                    json.dump(train_dict, fout)
                print("++++ Train Set Over ++++")    
            print("------- LOADING Train Set: " + str(count_1) + '/147671 -------')
        except:
            continue
        with open(train_json, 'w') as fout:
            json.dump(train_dict, fout)
    print("++++ Train Set Over ++++")         

def train_2_2(train_dict,count_1,file_list):
    ## train
    train_dict = dict(train_dict)
    count_1 = count_1.value
    for _file in file_list:
        if _file in train_dict.keys():
            print("~~~")
            continue
        try:
            file_name = _file.replace("\n","")
            with open(file_name, 'r') as fin:
                load_dict = json.load(fin)
            train_dict[file_name] = load_dict
            count_1 += 1
            if count_1 % 2 == 0:
                return 
            print("------- LOADING Train Set: " + str(count_1) + '/147671 -------')
        except:
            continue



def test_2(test_file_list, test_json):
    ## test
    count_1 = 0
    test_dict = {}
    test_set_count = len(test_file_list)
    for _file in test_file_list:
        try:
            file_name = _file.replace("\n","")
            with open(file_name, 'r') as fin:
                load_dict = json.load(fin)
            test_dict[file_name] = load_dict
            count_1 += 1
            print("------- LOADING Test Set: " + str(count_1) + '/ 18460 -------')
        except:
            continue    

    print("-> Now your TEST set has " + str(len(test_dict)) + " samples!")
    with open(test_json, 'w') as fout:
        json.dump(test_dict, fout)
    print("++++ Test Set Over ++++\n\n\n\n\n")


def valid_2(val_file_list, val_json):
    count_1 = 0
    val_dict = {}
    test_set_count = len(val_file_list)
    for _file in val_file_list:
        try:
            file_name = _file.replace("\n","")
            with open(file_name, 'r') as fin:
                load_dict = json.load(fin)
            val_dict[file_name] = load_dict
            count_1 += 1
            print("------- LOADING Val Set: " + str(count_1) + '/ 18460 -------')
        except:
            continue    

    print("-> Now your Val set has " + str(len(val_dict)) + " samples!")
    with open(val_json, 'w') as fout:
        json.dump(val_dict, fout)
    print("++++ Test Set Over ++++\n\n\n\n\n")
    
def valid_2_1(ns, _file):
    # val_dict = ns.df
    file_name = _file.replace("\n","")

    if file_name in val_dict.index.tolist():
        return
    try:
        val_dict.loc[file_name] = [1,2,3]
        # val_dict.loc[file_name] = []
        with open (file_name) as subgraph:
            json_text = json.load(subgraph)
            node_features = json_text["node_features"]
            graph = json_text["graph"]
            target = json_text["target"]
        val_dict.loc[file_name] = [node_features, graph, target]
        # val_dict.loc[file_name] = [1, 2, 3]
        print(val_dict.shape[0])
    except:
        return
     



def main():
    gnn_type = ["1", "2"]
    all_dataset_path = "/home/Devign-master/dataset/"
    dataset_path_list = ['/home/Devign-master/dataset/gnn1_test/', '/home/Devign-master/dataset/gnn2_input/']
    input_data = '/home/Devign-master/dataset/subgraph_json/'
    rec_txt = "/home/Devign-master/dataset/json_record.txt"

    
    gnn = '2'
    if gnn == gnn_type[0]:
        dataset_path = dataset_path_list[0]
    else:
        dataset_path = dataset_path_list[1]

    train_rec = all_dataset_path + 'train.txt' #记录已经分好的数据集
    test_rec = all_dataset_path + 'test.txt'
    val_rec = all_dataset_path + 'val.txt'

    train_json = "/home/Devign-master/dataset/gnn2_input/triain_GGNNinput/triain_GGNNinput_1.json"
    val_json = dataset_path + 'valid_GGNNinput.json'
    test_json = dataset_path + 'test_GGNNinput.json'
    
    if not os.path.exists(train_json):
        os.system("touch "+ train_json)
    if not os.path.exists(val_json):
        os.system("touch "+ val_json)
    if not os.path.exists(test_json):
        os.system("touch "+ test_json)

    test_set = set()
    val_set = set()
    train_set = set()
    
    with open(rec_txt, 'r') as f_rec:
        _file_list = f_rec.readlines() # 全部待处理的文件夹路径
    if _file_list == []: 
        sub_json_list = os.listdir(input_data)
        rec_json_list = []
        for json in sub_json_list:
            json += '\n'
            rec_json_list.append(input_data+json)
        with open("/home/Devign-master/dataset/json_record.txt", 'w') as f_rec:
            f_rec.writelines(rec_json_list)
            _file_list = rec_json_list
    try:
        with open(train_rec, 'r') as f_t: # 查看有没有分好的数据集
            train_list = f_t.readlines()
        with open(test_rec, 'r') as f_te:
            test_list = f_te.readlines()
        with open(val_rec, 'r') as f_r:
            val_list = f_r.readlines()
    except:
        split(test_set, val_set, train_set, _file_list, all_dataset_path)
        with open(train_rec, 'r') as f_t: # 记录分好的数据集
            train_list = f_t.readlines()
        with open(test_rec, 'r') as f_te:
            test_list = f_te.readlines()
        with open(val_rec, 'r') as f_r:
            val_list = f_r.readlines()        
    
    if gnn == gnn_type[0]:
        file_list = glob.glob(input_data+'*')
        split(test_set, val_set, train_set, file_list, dataset_path)
        print(len(test_set))
        train_1(train_set, file_list, train_json)
        val_test_1(test_set, val_set, file_list, test_json, val_json)
                   
    else:
        train_rec = dataset_path + 'train.txt' #记录已经分好的数据集
        test_rec = dataset_path + 'test.txt'
        val_rec = dataset_path + 'val.txt'

        try:
            with open(train_rec, 'r') as f_t: # 查看有没有分好的数据集
                train_list_2 = f_t.readlines()
            with open(test_rec, 'r') as f_te:
                test_list_2 = f_te.readlines()
            with open(val_rec, 'r') as f_r:
                val_list_2 = f_r.readlines()
            print(1)
        except:
            train_list_2 = []
            test_list_2 = []
            val_list_2 = []
            train_cnt = 0
            for path in train_list:
                path = path.replace("\n", "/")
                files = glob.glob(path+'*.json')
                for _file in files:
                    _file += '\n'
                    train_cnt += 1
                    print("Record train ----> " + str(train_cnt))
                    train_list_2.append(_file)
                print("!!!! " + str(len(train_list_2)))

            with open(train_rec, 'w') as f_t:
                f_t.writelines(train_list_2)

            val_cnt = 0
            for path in val_list:
                path = path.replace("\n", "/")
                files = glob.glob(path+'*.json')
                for _file in files:
                    _file += '\n'
                    val_cnt += 1
                    val_list_2.append(_file)
                    print("Record val ----> " + str(val_cnt))
                print("!!!! " + str(len(val_list_2)))
            with open(val_rec, 'w') as f_t:
                f_t.writelines(val_list_2)

            test_cnt = 0
            for path in test_list:
                path = path.replace("\n", "/")
                files = glob.glob(path+'*.json')
                for _file in files:
                    _file += '\n'
                    test_cnt += 1
                    test_list_2.append(_file)
                    print("Record test ----> " + str(test_cnt))
            with open(test_rec, 'w') as f_te:
                f_te.writelines(test_list_2)

            with open(train_rec, 'r') as f_t: # 查看有没有分好的数据集
                train_list_2 = f_t.readlines()
            with open(test_rec, 'r') as f_te:
                test_list_2 = f_te.readlines()
            with open(val_rec, 'r') as f_r:
                val_list_2 = f_r.readlines()     

        pool = Pool(1)
        m = multiprocessing.Manager()
        # train_dict = m.dict()
        ns = m.Namespace()
        json_data = pd.DataFrame(columns=["node_features", "graph", "target"])
        json_data.loc['1'] = [1,2,3]
        print(json_data)
        ns.df = json_data
        val_dict = ns.df
        pool.map(partial(valid_2_1, ns = ns), train_list_2[15000:30000])

        pool = Pool(1)
        m = multiprocessing.Manager()
        train_dict = m.dict()
        count_1 = m.Value('i',0) 

        p1 = multiprocessing.Process(target = train_2_2, args = (train_dict, count_1, train_list_2[15000:30000]))
        p2 = multiprocessing.Process(target = train_2_2, args = (train_dict, count_1, train_list_2[15000:30000]))
        p3 = multiprocessing.Process(target = train_2_2, args = (train_dict, count_1, train_list_2[15000:30000]))
        p4 = multiprocessing.Process(target = train_2_2, args = (train_dict, count_1, train_list_2[15000:30000]))
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        with open(train_json, 'w') as fout:
            json.dump(dic(train_dict), fout)
        print("++++ Train Set Over ++++")     
        


if __name__ == '__main__':
    main()