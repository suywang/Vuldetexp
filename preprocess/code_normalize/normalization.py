# coding=utf-8
import os
import re
import shutil
from clean_gadget import clean_gadget


# 预处理文件，每个C文件外包一层同名文件夹，并将文件放入joern的指定路径
def preprocess(org_path, joern_path):
    SARD = joern_path + "/20000"
    SARD_Vul = SARD + "/Vul"
    SARD_NoVul = SARD + "/NoVul"
    nvd = joern_path + "/ALL_9"
    nvd_Vul = nvd + "/Vul"
    nvd_NoVul = nvd +"/NoVul"
    if not os.path.exists(SARD):
        os.mkdir(SARD)
    if not os.path.exists(SARD_Vul):
        os.mkdir(SARD_Vul)
    if not os.path.exists(SARD_NoVul):
        os.mkdir(SARD_NoVul)
    if not os.path.exists(nvd):
        os.mkdir(nvd)
    if not os.path.exists(nvd_Vul):
        os.mkdir(nvd_Vul)
    if not os.path.exists(nvd_NoVul):
        os.mkdir(nvd_NoVul)

    setfolderlist = os.listdir(org_path)
    for setfolder in setfolderlist:
        catefolderlist = os.listdir(org_path + "/" + setfolder)
        for catefolder in catefolderlist:
            filelist = os.listdir(org_path + "/" + setfolder + "/" + catefolder)
            for file in filelist:
                filename = file[:-2]
                oldpath = org_path + "/" + setfolder + "/" + catefolder
                newpath = joern_path + "/" + setfolder + "/" + catefolder + "/" + filename
                if not os.path.exists(newpath):
                    os.mkdir(newpath)
                shutil.copy(oldpath + "/" + file, newpath)


# 遍历预处理后的文件，对每个文件进行规范化
def normalize(path):
    setfolderlist = os.listdir(path)
    file_list = os.listdir(path)
    for _file in file_list:
        pro_one_file(os.path.join(path, _file))
    '''for setfolder in setfolderlist:
        filepath = os.path.join(path, setfolder)
        file_list = os.listdir(filepath)
        filepath_tmp = filepath.replace("Vul", "sard-vul-src_without_comment")
        print("---=-=-=-=-=-=-=", len(os.listdir('/home/Final_Dataset_Old/sard-vul-src_without_comment')))
        if not os.path.exists(filepath_tmp):
            os.mkdir(filepath_tmp)
        else:
            continue
        for _file in file_list:
            pro_one_file(os.path.join(filepath, _file))'''


# 单个文件规范化
def pro_one_file(filepath):
    # 去除头部注释
    print(filepath)
    linefeed='\n'
    with open(filepath, "r") as file:
        code = file.read()
    annotations = re.findall('(?<!:)\\/\\/.*|\\/\\*(?:\\s|.)*?\\*\\/', code)
    #print(annotations)
    for annotation in annotations:
        lf_num = annotation.count('\n')
        if lf_num == 0:
            code = code.replace(annotation,'')
            continue
        code = code.replace(annotation,lf_num*linefeed)
    #code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', '', code)
    file_name=filepath.split('/')[-1]
    #filepath = filepath.replace("all_novul_slices_src", "8-slices-src_without_comment")
    temp_file='/home/mytest/temp'
    temp_file2=os.path.join(temp_file, file_name)
    #with open(filepath, "w") as file:
    with open(temp_file2, "w") as file:
        file.write(code.strip())
    # 规范化
    #with open(filepath, "r") as file:
    with open(temp_file2, "r") as file:
        org_code = file.readlines()
        print(org_code)
        print('\n\n\n\n')
        nor_code = clean_gadget(org_code)
    #filepath = filepath.replace("8-slices-src_without_comment", "9-norm-slices-src")
    #file_tmp = filepath.replace(filepath.rsplit("/")[-1], '')
    #if not os.path.exists(file_tmp):
        #os.mkdir(file_tmp)
    norm_path="/home/mytest/0day_normal"
    norm_path2=os.path.join(norm_path, file_name)
    #with open(filepath, "w") as file:
    with open(norm_path2, "w") as file:
        file.writelines(nor_code)


if __name__ == '__main__':
    # preprocess("/mnt/ysr/data_original", "/mnt/ysr/data")
    normalize("/home/mytest/0day")
    # pro_one_file("/mnt/ysr/data_ocriginal/basic-00001-min.c")
