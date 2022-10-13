import pandas as pd
import os
import difflib
import pickle

def process_raw_data(filter_csv_path, raw_csv_path):
    if os.path.exists(filter_csv_path):
        pd_filter = pd.read_csv(filter_csv_path)
        # pd_filter = pd.read_csv('/home/MSR_data_cleaned_pairs_Test.csv')

    else:
        pd_raw = pd.read_csv(raw_csv_path)
        filter_csv = pd_raw.loc[pd_raw["vul"] == 1]
        filter_csv.to_csv(filter_csv_path)
        pd_filter = pd.read_csv(filter_csv_path)
    return pd_filter

def label(f_vul, f_novul ,label_dict, outfile):
    diff = list(difflib.unified_diff(f_novul.splitlines(), f_vul.splitlines()))
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
            elif line.startswith("+"):
                line_num -= 1
            line_num += 1
        i += 1
        

def main():
    dataset_path = '/home/Devign-master/dataset/'
    raw_data_path = 'raw_data/'
    raw_data_filename = 'MSR_data_cleaned.csv'
    filter_data_filename = 'MSR_data_filtered.csv'

    dataset_path_output = 'dataset_test/'
    label_pkl_file = 'test_label_pkl.pkl'
    
    filter_csv_path = dataset_path + raw_data_path + filter_data_filename
    raw_csv_path = dataset_path + raw_data_path + raw_data_filename
    output_path = dataset_path + dataset_path_output
    pkl_path = dataset_path + label_pkl_file

    pd_filter = process_raw_data(filter_csv_path, raw_csv_path)
    file_cnt = pd_filter.shape[0]
    file_num = 0
    label_dict = {}
    cnt_1 = 0

    for index, row in pd_filter.iterrows():
        file_num += 1
        print(str(file_cnt) + ' / ' + str(file_num))
        project_name = row["project"]
        hash_vaule = row['commit_id']
        file_name = project_name + "_" + hash_vaule
        outfile = output_path + file_name

        file_name_cnt = 0
        outfile_new = outfile
        while outfile_new in label_dict.keys():
            outfile_new = outfile + '_' + str(file_name_cnt)
            file_name_cnt += 1

        if not os.path.exists(outfile_new):
            os.mkdir(outfile_new)

        label_dict[outfile_new] = []

        func_before = row['func_before']
        func_after = row["func_after"]
        vul_file_name = '1_'+ file_name + '.c'
        novul_file_name = '0_' + file_name+ '.c'

        with open(outfile_new + '/'+ vul_file_name, 'w', encoding='utf-8') as f_vul:
            f_vul.write(func_before)
            cnt_1 += 1

        with open(outfile_new + '/' + novul_file_name, 'w', encoding='utf-8') as f_novul:
            f_novul.write(func_after)
            cnt_1 += 1

        if pd.isnull(row['lines_before']):
            label_dict[outfile_new] = ['']
        else:
            label(func_before, func_after, label_dict, outfile_new)

    with open(pkl_path,'wb') as f:
        pickle.dump(label_dict, f)

    print(cnt_1)


if __name__ == '__main__':
    main()