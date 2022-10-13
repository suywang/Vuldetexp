# -- coding: utf-8 --
from access_db_operate import *
# from complete_PDG import *
import re
from py2neo.packages.httpstream import http
import subprocess
import time

http.socket_timeout = 9999

# def get_cpg(node, cpgpath):
#     if not os.path.exists(cpgpath):
#         os.mkdir(cpgpath)
#     cmd = 'echo ' + str(node._id) + ' | joern-plot-proggraph -all -a code type > ' + cpgpath + "/" + cpgpath.split("/")[-1] + ".dot"
#     print cmd
#     subprocess.call(cmd,shell=True)
def get_cpg(file):
    with open(file, "r") as file:
        linelist = file.readlines()
        for line in linelist:
            line = line.strip()
            id = line.split(" ")[0]
            path = line.split("/")[-1]
            cmd = 'echo ' + str(id) + ' | joern-plot-proggraph -all -a code type > ' + path + '.dot'
            while True:
                process = len(os.popen('ps aux | grep "' + "joern-plot-prog" + '" | grep -v grep | grep -v tail | grep -v keepH5ssAlive').readlines())
                print process
                if process==0:
                    break
            print "\n" + cmd
            subprocess.call(cmd,shell=True)


if __name__ == '__main__':
    get_cpg("/mnt/ysr/cpg/id-path.txt")
