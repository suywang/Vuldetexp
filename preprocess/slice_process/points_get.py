import os
import re
import pickle,json


def get_gt_node(node_dict,dict_path,name,label_path):
    pointer_node_list = []
    identifier_list = []
    with open(label_path,'r') as lp:
       vul_line=json.load(lp)
    try:
        # line_label = vul_line[name[2:]]
        line_label = vul_line[name[2:]+'.c']
    except:
        line_label = []
    if line_label == []:
        return []
    func_name = name.split('/')[-1]
    dict_name = func_name+'.json'
    with open(dict_path+dict_name,'r') as rf1:
        try:
            node2line_dict=json.load(rf1)
        except:
            return []
    for node in node_dict:
        node_type = node_dict[node].node_type
        node_num = node_dict[node].id.split('id=')[-1].split(']')[0]
        for node_x in node2line_dict:
            if node_x['node'] == node_num:
                if int(node_x['line_num']) in line_label:
                    pointer_node_list.append(node_dict[node])
                break
    pointer_node_list = list(set(pointer_node_list))
    return pointer_node_list




def get_pointers_node(node_dict):
    pointer_node_list = []
    identifier_list = []
    identifier_node_type = ['Identifier', 'MethodParameterIn', 'FieldIdentifier']
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type in identifier_node_type:
            identifier_list.append(node_dict[node])
    for node in identifier_list:
        node_code = node.properties.code()
        indx_1 = node_code.find("*")
        node_has_full_type = node.properties.has_type()
        if node_has_full_type != False:
            node_full_type = node.properties.get_type()
            indx_2 = node_full_type.find("*")
        else:
            indx_2 = -1
       
        if indx_1 != -1 or indx_2 != -1:
            pointer_node_list.append(node)
    pointer_node_list = list(set(pointer_node_list))
    return pointer_node_list


def get_all_array(node_dict):
    array_node_list = []
    identifier_list = []
    identifier_node_type = ['indirectIndexAccess', 'MethodParameterIn']
    for node in node_dict:
        node_type = node_dict[node].label
        if node_type in identifier_node_type:
            identifier_list.append(node_dict[node])
    for node in identifier_list:
        node_code = node.properties.code()
        if node_code.find("[") != -1:
            array_node_list.append(node)
    array_node_list = list(set(array_node_list))
    return array_node_list
    
def get_all_sensitiveAPI(node_dict):
    with open("/home/mytest/slice/sensitive_func.pkl", "rb") as fin:
        list_sensitive_funcname = pickle.load(fin)
    call_node_list = []
    call_type = "Call"   
    for func_name in list_sensitive_funcname:
        for node in node_dict:
            node_type = node_dict[node].node_type
            node_code = node_dict[node].properties.code().split("(")[0]
            if node_type == call_type:
                if func_name in node_code:
                    call_node_list.append(node_dict[node])
    return call_node_list

def get_all_integeroverflow_point(node_dict):
    interoverflow_list = []
    exp_node_list = []
    exp_type = 'assignment'
    for node in node_dict:
        node_type = node_dict[node].label
        if node_type == exp_type:
            exp_node_list.append(node)
    for node in exp_node_list:
        node_code = node_dict[node].properties.code()
        # print(node_code)
        if node_code.find("="):
            code = node_code.split('=')[-1].strip()
            pattern = re.compile("((?:_|[A-Za-z])\w*(?:\s(?:\+|\-|\*|\/)\s(?:_|[A-Za-z])\w*)+)")
        else:
            code = node_code
            pattern = re.compile("(?:\s\/\s(?:_|[A-Za-z])\w*\s)")
        results = re.search(pattern, code)
        if results != None:
            interoverflow_list.append(node)
    return interoverflow_list