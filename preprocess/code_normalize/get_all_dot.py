# -- coding: utf-8 --
from access_db_operate import *
# from complete_PDG import *
import re
from py2neo.packages.httpstream import http
import subprocess

http.socket_timeout = 9999

def get_cpg(node, cpgpath):
    if not os.path.exists(cpgpath):
        os.mkdir(cpgpath)
    cmd = 'echo ' + str(node._id) + ' | joern-plot-proggraph -all -a code type > ' + cpgpath + "/" + cpgpath.split("/")[-1] + ".dot"
    print cmd
    subprocess.call(cmd,shell=True)

#  生成全部dot，保存到一个文件中
def get_all_dot(funcfile, file):
    with open(funcfile, "r") as funcfile:
        funcname = funcfile.readlines()
    funcfile.close()
    for name in funcname:
        name = name.strip()
        print "funcname:",name
        cmd = 'echo \'getFunctionsByName("'+name+ '").id\' | joern-lookup -g | joern-plot-proggraph -all -a code type >> ' + file
        subprocess.call(cmd,shell=True)


if __name__ == '__main__':
    get_all_dot("/mnt/ysr/cpg/funcname.txt","/mnt/ysr/cpg/all.dot")
