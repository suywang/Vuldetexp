import os, sys
import re
import json
import glob
sys.path.append("/home/mVulPreter")
from utils_dataset.objects.cpg.function import Function
from collections import OrderedDict
from slice.points_get import *


def graph_indexing(graph):
    idx = graph["file"].split(".c")[0].split("/")[-1]
    del graph["file"]
    return idx, {"functions": [graph]}

def joern_process(in_path):
    if in_path[-1] != '/':
        in_path += '/'
    files = glob.glob(in_path + '*.json')
    container = []
    for file in files:
        if os.path.exists(file):
            with open(file,'r',encoding='utf-8') as jf:
                cpg_string = jf.read()
                cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)
                cpg_json = json.loads(cpg_string)
                container.append([graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"])
    return container    


def filter_nodes(nodes):
    return {n_id: node for n_id, node in nodes.items() if node.has_code() and
            node.has_line_number() and
            node.label not in ["Comment", "Unknown"]}

def order_nodes(nodes):
    # sorts nodes by line and column
    nodes_by_column = sorted(nodes.items(), key=lambda n: n[1].get_column_number())
    nodes_by_line = sorted(nodes_by_column, key=lambda n: n[1].get_line_number())

    for i, node in enumerate(nodes_by_line):
        node[1].order = i

    return OrderedDict(nodes_by_line)


def parse_to_nodes(cpg):
    nodes = {}
    for function in cpg["functions"]:
        func = Function(function)
        node_tmp = func.get_nodes()
        # Only nodes with code and line number are selected
        filtered_nodes = filter_nodes(node_tmp)
        nodes.update(filtered_nodes)
    return order_nodes(nodes)



