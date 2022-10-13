import os, sys
import glob
import shutil
import networkx as nx
import json
import glob
from tqdm import tqdm

 
if __name__ == "__main__":
  #dot_files="/home/mytest/nvd/all_slices"
  with open('/home/mytest/0day/dataset/Interpretation.txt','r') as fp:
    dot_files = fp.readlines()
  files=[]
  for dot in dot_files:
    name=dot.split('/')[-1].split('.json')[0]+'.dot'
    files.append('/home/mytest/0day/all_slices/'+name)
  lineinfo_files="/home/mytest/0day/0day_lineinfo/"
  #files=glob.glob(dot_files+'/*')
  #log_path="/home/mytest/log.txt"
  node_index=dict()
  cnt=1
  for file in tqdm(files):
    name=file.split('/')[-1].split('@')[0]
    true_name=file.split('/')[-1]
    if true_name.startswith('1_'):
      if os.path.exists('/home/mytest/0day/0day_linedict'+true_name+'.json'):
        print(true_name+' has been processed...')
        print(cnt)
        cnt+=1
        continue
      all_res=[]
      try:
        pdg = nx.drawing.nx_pydot.read_dot(file)
        if type(pdg) != None:
          for index, node in enumerate(pdg.nodes()):
            if node!='\\n':
              node_index[node] = index
          #with open(lineinfo_files+'1_'+name[2:]+'.json','r') as fp:
          with open(lineinfo_files+name[2:]+'.json','r') as fp:
            line_dict=json.load(fp)
          for method_func in line_dict['functions']:
            for id in method_func['AST']:
              real_id=id['id'].split('id=')[-1].split(']')[0]
              for propertie in id['properties']:
                if propertie['key']=='LINE_NUMBER':
                  #res.append(node,node_index[node],propertie['value'])
                  if real_id in node_index:
                    res={}
                    res['node']=real_id
                    res['node_id']=node_index[real_id]
                    res['line_num']=propertie['value']
                    all_res.append(res)
          with open("/home/mytest/0day/0day_linedict/"+true_name+'.json','w') as fp:
            json.dump(all_res,fp)
            #print(cnt)
            cnt+=1
      except:
        continue
