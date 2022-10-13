import glob
import networkx as nx
import os

src_file='/home/mytest/0day/all_slices'
files = glob.glob(src_file+'/*')
cnt=1
for file in files:
    with open(file,'r+') as fp:
        flists=fp.readlines()
    #fp.close()
    temp=[]
    flag=0
    for flist in flists:
        if'\\' in flist and '->' in flist:
            flag=1
            modify_line=flist.split('[label')[0]+'[label="DDG: "];\n'
            temp.append(modify_line)
        #if flist == '\n':
        else:
            temp.append(flist)
    with open(file,'w+') as wf:
        wf.writelines(temp)
    #wf.close()
    if flag == 1:
        print(cnt)
        cnt+=1
    #try:
    #    pdg = nx.drawing.nx_pydot.read_dot(file)
    #    flag=1
    #except:
    #    print('error')
    #    continue
    #if flag == 1:
    #    print(cnt)
    #    cnt+=1
    #print(pdg.nodes())