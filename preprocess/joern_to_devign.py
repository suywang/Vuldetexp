import networkx as nx
from gensim.models import KeyedVectors
import warnings
import argparse
import glob
from multiprocessing import Pool
from functools import partial
import numpy as np
import json
import os

warnings.filterwarnings("ignore")
cnt = 0

def tokenize_code_line(line):
    # Sets for operators
    operators3 = {'<<=', '>>='}
    operators2 = {
        '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
        '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
    operators1 = {
        '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
        '=', ',', '?', ':', ';', '{', '}', '!', '~'
    }

    tmp, w = [], []
    i = 0
    if type(i) == None:
        return []
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 3].strip())
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 2].strip())
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    if (len(w) != 0):
        tmp.append(''.join(w).strip())
        w = []
    # Filter out irrelevant strings
    tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
    return tmp

def joern_to_devign(dot_pdg, word_vectors, out_path):
    out_path = out_path + dot_pdg.split("/")[3]+'/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    name = dot_pdg.split('/')[-1].split('.')[0]
    out_json = out_path + name + '.json'
    if os.path.exists(out_json):
        print("-----> has been processed :\t", out_json)
        return
    print("===============\t"+dot_pdg)
    vul = int(name.split('_')[0])
    node_index = dict()
    node_feature = dict()
    try:
        pdg = nx.drawing.nx_pydot.read_dot(dot_pdg)

        if type(pdg) != None:
            for index, node in enumerate(pdg.nodes()):
                node_index[node] = index
                label = pdg.nodes[node]['label'][1:-1]
                code = label.partition(',')[2]
                feature = np.array([0.0 for i in range(100)])
                for token in tokenize_code_line(code):
                    if token in word_vectors:
                        feature += np.array(word_vectors[token])
                    else:
                        feature += np.array([0.0 for i in range(100)])
                node_feature[index] = feature

            nodes_ = []
            for i in range(len(list(pdg.nodes()))):
                nodes_.append(list(node_feature[i]))

            edges_ = []
            for item in pdg.adj.items():
                s = item[0]
                for edge_relation in item[1]:
                    d = edge_relation    
                    ddg_flag = 0
                    cdg_flag = 0 
                    for edge in item[1]._atlas[edge_relation].items():
                        if 'DDG' in edge[1]['label'] and ddg_flag == 0:
                            edge_type = 0
                            ddg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))
                        elif 'CDG' in edge[1]['label'] and cdg_flag == 0:
                            edge_type = 1
                            cdg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))

            data = dict()
            data['node_features'] = nodes_
            data['graph'] = edges_
            data['target'] = vul
            out_json = out_path + name + '.json'
            with open(out_json, 'w') as f:
                f.write(json.dumps(data))  

    except:
        pass
    return 

def main():
    '''dir_path_list = ['/home/joern_res_pdg/ffmpeg_novul/', '/home/joern_res_pdg/ffmpeg_vul/',
    '/home/joern_res_pdg/qemu_novul/', '/home/joern_res_pdg/qemu_vul/']
    out_path = '/home/devign_pdg_new/'
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/home/test1/')
    parser.add_argument('--output_dir', type=str, help='Output Directory of the parser',default='/home/test2/')
    args = parser.parse_args()

    dir_path_list = [args.input_dir]
    out_path = args.output_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    dots = []
    for dir_path in dir_path_list:
        dots_tmp = glob.glob(dir_path + '*.dot')
        for dot in dots_tmp:
            if '-' in dot.rsplit('/')[-1]: #说明是从joern直接解析的结果，没有重新命名过
                if 'novul' in dir_path: #无漏洞的为0_开头
                    new_name = dot[:dot.rindex('/')+1] + '0_' + dot[dot.rindex('/')+1:].replace("-pdg","")
                else: # 有漏洞的为1_开头
                    new_name = dot[:dot.rindex('/')+1] + '1_' + dot[dot.rindex('/')+1:].replace("-pdg","")
                os.system("mv "+ dot + ' ' + new_name)
            else:
                new_name = dot
            dots.append(new_name)
    #读取词向量模型w2v
    word_vectors = KeyedVectors.load('/home/mVulPreter/w2v.wv', mmap='r')
    pool = Pool(4)
    pool.map(partial(joern_to_devign, word_vectors=word_vectors, out_path=out_path), dots)

if __name__ == '__main__':
    main()

