# -- coding: utf-8 --
from access_db_operate import *
# from complete_PDG import *
import re
from py2neo.packages.httpstream import http
import subprocess

http.socket_timeout = 9999
funcname = set()


#  生成全部dot，保存到一个文件中
def get_all_dot(file):
    for name in funcname:
        print "funcname:",name
        cmd = 'echo \'getFunctionsByName("'+name+ '").id\' | joern-lookup -g | joern-plot-proggraph -all -a code type >> ' + file
        subprocess.call(cmd,shell=True)


#  将"函数id 函数路径"，保存到文件中
#  将"函数名"保存到文件
def get_id_path(id_file, funcfile):
    j = JoernSteps()
    j.connectToDatabase()
    all_func_node = getALLFuncNode(j)
    with open(id_file,"a") as id_file:
        for node in all_func_node:
            funcname.add(node.properties['name'])
            # print "\n"
            # print "node.id:",node._id
            # print "node.properties['name']:",node.properties['name']
            list = getFuncFile(j, node._id).split('/')
            # if not os.path.exists(list[-4]):
            #     os.mkdir(list[-4])
            # if not os.path.exists(list[-4] + "/" + list[-3]):
            #     os.mkdir(list[-4] + "/" + list[-3])
            dot_path = list[-4] + "/" + list[-3] + "/" + list[-2]
            # print dot_path
            # get_cpg(node, dot_path)
            id_file.writelines(str(node._id) + " " + dot_path + "\n")
    id_file.close()
    with open(funcfile,"a") as funcfile:
        for name in funcname:
            funcfile.write(name + "\n")
    funcfile.close()
    print ("funcname:",funcname)


if __name__ == '__main__':
    get_id_path("/mnt/ysr/cpg/id-path.txt", "/mnt/ysr/cpg/funcname.txt")
