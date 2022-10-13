from argparse import ArgumentParser
from typing import cast, List
from omegaconf import OmegaConf, DictConfig
import json
import glob
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
from os import cpu_count
from tqdm import tqdm
from multiprocessing import cpu_count, Manager, Pool
import functools
import os
# from src.utils import PAD, MASK, UNK
PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
# SPECIAL_TOKENS = [PAD, UNK, MASK]
USE_CPU = cpu_count()

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


def process_parallel(path: str, split_token: bool):
    """

    Args:
        path:

    Returns:

    """
    #xfg = nx.read_gpickle(path)
    node_index = dict()
    tokens_list = list()
    try:
        pdg = nx.drawing.nx_pydot.read_dot(path)
        for index, node in enumerate(pdg.nodes()):
                node_index[node] = index
                try:
                    label = pdg.nodes[node]['label'][1:-1]
                except:
                    continue
                code = label.partition(',')[2]
                for token in tokenize_code_line(code):
                    tokens_list.append(token)
    except:
        pass
    #for ln in xfg:
        #code_tokens = xfg.nodes[ln]["code_sym_token"]

        #if len(code_tokens) != 0:
            #tokens_list.append(code_tokens)

    return tokens_list


def train_word_embedding(config_path: str):
    """
    train word embedding using word2vec

    Args:
        config_path:

    Returns:

    """
    # config = cast(DictConfig, config_path.strip())
    config = cast(DictConfig, OmegaConf.load(config_path))
    cweid = config.dataset.name
    root = config.data_folder
    #train_json = f"{root}/{cweid}/train.json"
    train_path = "/home/mytest/nvd/all_slices"
    #with open(train_json, "r") as f:
        #paths = json.load(f)
    paths=glob.glob(train_path+'/*')
    tokens_list = list()
    with Manager():
        pool = Pool(USE_CPU)

        process_func = functools.partial(process_parallel,
                                         split_token=config.split_token)
        tokens: List = [
            res
            for res in tqdm(
                pool.imap_unordered(process_func, paths),
                desc=f"pdg paths: ",
                total=len(paths),
            )
        ]
        pool.close()
        pool.join()
    cnt=0
    for token in tokens:
        if token == []:
            cnt+=1
    print(cnt)
    for token_l in tokens:
        tokens_list.extend(token_l)
    print("training w2v...")
    print(len(tokens_list))
    num_workers = cpu_count(
    ) if config.num_workers == -1 else config.num_workers
    model = Word2Vec(sentences=tokens_list, min_count=3, size=256,#vector_size=config.gnn.embed_size,
                     max_vocab_size=config.dataset.token.vocabulary_size, workers=num_workers, sg=1)
    model.wv.save("/home/deepwukong/w2v.wv")


def load_wv(config_path: str):
    """

    Args:
        config_path:

    Returns:

    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    cweid = config.dataset.name

    model = KeyedVectors.load(f"{config.data_folder}/{cweid}/w2v.wv", mmap="r")

    print()


if __name__ == '__main__':
    os.chdir("/home/mytest")
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c",
                              "--config",
                              help="Path to YAML configuration file",
                              default="configs/config.yaml",
                              type=str)
    __args = __arg_parser.parse_args()
    train_word_embedding(__args.config)
    # load_wv(__args.config)
