# -- coding: utf-8 --
from access_db_operate import *
# from complete_PDG import *
import re
from py2neo.packages.httpstream import http
import subprocess

http.socket_timeout = 9999
funcname = set()

def get_cpg(node, cpgpath):
    if not os.path.exists(cpgpath):
        os.mkdir(cpgpath)
    cmd = 'echo ' + str(node._id) + ' | joern-plot-proggraph -all -a code type > ' + cpgpath + "/" + cpgpath.split("/")[-1] + ".dot"
    print cmd
    subprocess.call(cmd,shell=True)

#  生成全部dot，保存到一个文件中
def get_all_dot(file):
    for name in funcname:
        print ("funcname:",name)
        cmd = 'echo \'getFunctionsByName("'+name+ '").id\' | joern-lookup -g | joern-plot-proggraph -all -a code type >> ' + file
        subprocess.call(cmd,shell=True)

def get_single_dot(file):
    with open(file,"r") as file:
        linelist = file.readlines()
        if not os.path.exists("single"):
            os.mkdir("single")
        for line in linelist:
            if line[0:2]=="//" and line[2:].strip().isdigit():
                # print line
                dot_name = "single/" + line[2:].strip() + ".dot"
                print (dot_name)
                single_dot = ""
            single_dot = single_dot + line
            if line.strip() == "//###":
                # print single_dot
                with open(dot_name, "w") as singlefile:
                    singlefile.write(single_dot)
                singlefile.close()
    file.close()


if __name__ == '__main__':
    get_single_dot("/mnt/ysr/cpg/all.dot")
