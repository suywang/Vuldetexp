import json
import os
import glob

#from nbformat import write


if __name__ == "__main__":

    code_paths="/home/nvd_csv/beforecode/"
    patch_paths="/home/nvd_csv/patch/"
    files = glob.glob(patch_paths+'/*')
    del_dict={}
    for file in files:
        name=file.split('/')[-1]
        #del_dict['name']=[]
        pf = open(file, 'r')
        del_num=0
        patch_list=pf.readlines()
        cf=open(code_paths+name,'r')
        code_list=cf.readlines()
        diff_line=0
        line_num=0
        diff_list=[]
        diff_code=[]
        for line in patch_list:
            if line.startswith('@@'):
                diff_list.append(line_num)
            line_num+=1
        for i,diffx  in  enumerate(diff_list):
            if i+1 < len(diff_list):
                diff_code.append(patch_list[diffx+1:diff_list[i+1]])
            else:
                diff_code.append(patch_list[diffx+1:])
        ground_truth=[]
        for del_line in diff_code:
            first=0
            #if del_line[0].startwith('-'):
                #del_line[0][0]=' '
            while True:
                try:
                    subground_truth=[]
                    findex=code_list.index(del_line[0],first)
                    first=findex+1
                    find_flag=True
                    for j,line in enumerate(del_line[1:]):
                        if line.startswith('-'):
                            line = line[1:]
                            real_line=''.join(line.split())
                            if not real_line.startswith('\n') and not real_line.startswith('//') and not real_line.startswith('/*') and not real_line.startswith('*/') and real_line!='':
                                subground_truth.append(findex+j+2)
                        if line!=code_list[findex+j+1]:
                            find_flag=False
                            break
                    if find_flag==True:
                        ground_truth.extend(subground_truth)
                except:
                    break
        if ground_truth!=[]:
            del_dict[name]=list(set(ground_truth))
    with open("/home/nvd_csv/nvd_vul_lineinfo.json",'w') as wp:
        json.dump(del_dict,wp)
