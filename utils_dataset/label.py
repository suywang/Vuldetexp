import os
import json
import pickle
import difflib

def record(file, no_vul_file, vul_file, vul, no_vul):
    check_path = '/home/Devign-master/dataset/check_tmp/'
    if not os.path.exists(check_path+file):
        os.mkdir(check_path+file)
    check_file = check_path +file +'/'+vul_file
    with open(check_file, 'w') as f_check:
        f_check.writelines(vul)
    check_file_2 = check_path +file +'/'+no_vul_file
    with open(check_file_2, 'w') as f_check:
        f_check.writelines(no_vul)            
    print(file)
    print('===============')


def label(diff,label_dict, outfile):
    list_tmp = []
    # diff = list(difflib.unified_diff(f_novul.splitlines(), f_vul.splitlines()))
    split_list = [i for i,line in enumerate(diff) if line.startswith("@@")]
    split_list.append(len(diff))
    i = 0
    for i in range(len(split_list) - 1):
        start = split_list[i]
        del_linenum = diff[start].split("@@ -")[-1].split(",")[0].split('+')[-1].strip()
        end = split_list[i + 1]
        
        line_num = int(del_linenum)
        for line in diff[start+1 : end]:
            if line.startswith("-"):
                label_dict[outfile].append(line_num)
                list_tmp.append(line_num)
            elif line.startswith("+"):
                line_num -= 1
            line_num += 1
        i += 1
    return list_tmp

def main():
    label_new = {}
    with open('/home/Devign-master/dataset/label.pkl', 'rb') as f_label:
        label_dict = pickle.load(f_label)
    path = '/home/Devign-master/dataset/dataset_vul_novul_patch/'
    i = 0
    null = 0
    null_old = 0
    cnt = 0
    for file in label_dict.keys():
        i+=1
        print("-------> " + str(i))
        full_path = path+file[2:]
        file_list = os.listdir(full_path)
        for _file in file_list:
            if _file.startswith("0_"):
                novul_name = _file
            else:
                vul_name = _file
        novul_file = full_path+'/'+novul_name
        vul_file = full_path+'/'+vul_name
        if not os.path.exists(novul_file):
            continue
        if not os.path.exists(vul_file):
            continue
        with open(novul_file,'r',encoding='utf-8') as fno:
            no_vul = fno.readlines()
        with open(vul_file,'r',encoding='utf-8') as fvul:
            vul = fvul.readlines()
        label_line = label_dict[file]
        if label_line == ['']:
            null_old += 1
        diff = list(difflib.unified_diff(vul,no_vul))
        del_flag = 0
        for line in diff:
            if line.startswith('-') and not line.startswith("---"):
                del_flag = 1
        if del_flag == 0:
            label_new[file] = ['']
            null += 1
            continue
        del_list = []
        for line in diff:###
            if line.startswith("@@ "):
                iter = 0
                add_del = line[3:].split("@@")[0].strip().split(" ")
                for changed in add_del:
                    if '-' in changed:
                        del_line_num = int(changed.split(',')[0][1:]) - 1
            elif line.startswith('+') and not line.startswith("+++"):
                continue
            elif line.startswith('+++') or line.startswith("---"):
                continue
            else:
                iter += 1
                if line.startswith('-') and not line.startswith("---"):
                    del_flag = 1
                    del_list.append(del_line_num+iter)
        if del_list == []:
            print(file)
        label_new[file] = del_list
        # label_line = label_dict[file]
        # if label_line == ['']:
        #     null_old += 1
        if del_list != label_line:
            # # record(file, novul_name, vul_name, vul, no_vul)
            # list_tmp = label(diff, label_dict, file)
            # if del_list != list_tmp:
            cnt += 1

    print('difference: ', cnt)
    print('new: ',null)
    print("old: ", null_old)
    with open("/home/Devign-master/dataset/func_label.pkl","wb") as f_pkl:
        pickle.dump(label_new, f_pkl)

        # if del_list != label_line:
        #     # record(file, novul_name, vul_name, vul, no_vul)
        #     list_tmp = label(diff, label_dict, file)
        #     if del_list == list_tmp:
        #         print(111)
        #     else:
        #         print(222)
        #     print(file)
        # if label_line != ['']:
        #     for line in diff[:]:
        #         if line.startswith("-"):
        #             diff.remove(line)
        #     for line in label_line:
        #         print(line)
        #         if line-1 >= len(vul):
        #             record(file, novul_name, vul_name, vul, no_vul)
        #             print(file)         
        #         else:
        #             line_code = vul[line - 1]
        #             if line_code.strip() == '\n':
        #                 continue
        #             if line_code in no_vul:
        #                 if line_code in diff:
        #                     record(file,novul_name, vul_name, vul, no_vul)          
        #                     print(vul[line - 1])
        #                     print(file)
        #                     print('===============')
        # else:
            
        #     for diff_line in diff[3:]:
        #         if diff_line.startswith("-"):
        #             record(file, novul_name, vul_name, vul, no_vul)       
        #             print(diff_line)
        #             print(file)
        #             print('===============')
        # # # d = difflib.Differ()
        # # diff_2 = d.compare(no_vul,vul)
        # # string = '\n'.join(diff_2)
        # # diff_list = string.splitlines()
        # line_num = 0
        # label_list = []
        # for line in diff[3:]:
        #     if not line.startswith("+"):
        #         line_num += 1 
        #         if line.startswith("-"):
        #             label_list.append(line_num)
        # if label_list == []:
        #     label_list = ['']
        # if label_list != label_line:
        #     check_path = '/home/Devign-master/dataset/check_tmp/'
            
        #     os.mkdir(check_path+file)
        #     check_file = check_path +file +'/'+file_list[1]
        #     with open(check_file, 'w') as f_check:
        #         f_check.writelines(vul)
        #     check_file_2 = check_path +file +'/'+file_list[0]
        #     with open(check_file_2, 'w') as f_check:
        #         f_check.writelines(no_vul)            
        #     print("Error")

        # print(1)

    #     print(label_dict[file])
    #     if label_dict[file] == ['']:
    #         i += 1
    # print(i)
    # print(len(label_dict))
    # for file_name_old in label_dict.keys():
    #     file_name_new = "1_"+file_name_old.rsplit("/")[-1]
    #     dict_new[file_name_new] = label_dict[file_name_old]
    # with open('/home/Devign-master/dataset/label.pkl', 'wb') as fin:
    #     pickle.dump(dict_new, fin)

    # record_path = '/home/Devign-master/dataset/record.txt'
    # json_path = '/home/Devign-master/dataset/joern_json/'
    # json_path_list = os.listdir(json_path)

    # json_pair = {}
    # single = []
    # i = 0
    # for path in json_path_list:
    #     if path.startswith("0_"):
    #         json_pair[path] = ''
    # for path in json_path_list:
    #     if path.startswith("1_"):
    #         novul_path = "0_" + path[2:]
    #         if novul_path in json_pair.keys():
    #             json_pair[novul_path] = path
    #         else:
    #             single.append(path)
    # for key in json_pair.keys():
    #     if json_pair[key] == '':
    #         single.append(key)
    # single = list(set(single))
    # string = ''
    # for path in single:
    #     path += '\n'
    #     string += path
    # with open("/home/Devign-master/dataset/record.txt",'w') as f:
    #     f.write(string)

    # print(len(path))
    

        
    #     full_path = json_path + path
    #     json_file_list.append(path)
    #     # with open(full_path, 'r') as f:
    #     #     i += 1
    #     #     try:
    #     #         data = json.load(f)
    #     #     except:
    #     #         print(full_path)
    #     #     print(str(i) + ' / ' + str(len(json_path_list)))
    #     #     json_file_list.append(path)
    
    # parse_res = '/home/Devign-master/dataset/joern_parse_res/'
    # remain_res = '/home/Devign-master/dataset/tmp_parse_res'
    # parse_path_list = os.listdir(parse_res)

    # record_list = []
    # for path in parse_path_list:
    #     parse_full_path = parse_res + path
    #     json_name = path[:-4] + '.json'
    #     if json_name not in json_file_list:
    #         record_list.append(path+'\n')
    
    # record_txt = ''
    # for record in record_list:
    #     record_txt += record

    # with open(record_path, 'w') as fw:
    #     fw.write(record_txt)

if __name__ == '__main__':
    main()