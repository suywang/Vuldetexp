import os, sys
import glob
import argparse
from multiprocessing import Pool
from functools import partial
import subprocess

def get_all_file(path):
    path = path[0]
    file_list = []
    path_list = os.listdir(path)
    for path_tmp in path_list:
        full = path + path_tmp + '/'
        for file in os.listdir(full):
            file_list.append(file)
    return file_list

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-i', '--input', help='A txt file including all path of targeted src files', type=str,default='/home/all_data_preprocess/3-src_norm/our_data/vul1/')
    parser.add_argument('-o', '--output', help='The dir path of output', type=str,default='/home/all_data_preprocess/4-parse_res/our_data/vul')
    parser.add_argument('-t', '--type', help='The type of procedures: parse or export', type=str,default='parse')
    args = parser.parse_args()
    return args

def joern_parse(file, outdir):
    record_txt =  os.path.join(outdir,"parse_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch "+record_txt)
    with open(record_txt,'r') as f:
        rec_list = f.readlines()
    file = file.strip()
    name = file.split('/')[-1][:-2]
    if name+'\n' in rec_list:
        print(" ====> has been processed: ", name)
        return
    print(' ----> now processing: ',name)
    out = outdir + name + '.bin'
    if os.path.exists(out):
        return
    os.environ['file'] = str(file)
    os.environ['out'] = str(out) #parse后的文件名与source文件名称一致
    os.system('sh joern-parse $file --language c --out $out')
    with open(record_txt, 'a+') as f:
        f.writelines(name+'\n')

def joern_export(bin, repr, outdir):
    bin = bin.strip()
    if repr == 'ddg':
        out = os.path.join(outdir,"3-ddg-nvd")
    else:
        out = os.path.join(outdir,"4-json-nvd")
    out += '/'
    try:
        if not os.path.exists(out):
            print(out)
            os.makedirs(out)
    except:
        pass
    name = bin.split('/')[-1][:-4]
    out += name
    print('+++++++++++++ repr:\t' + repr + '\t' +  out)
    os.environ['bin'] = str(bin)
    os.environ['out'] = str(out)
    if os.path.exists(out):
        print("================== has been procesed:\t"+out)
        return

    if repr == 'ddg':
        os.system('sh joern-export $bin'+ " --repr " + repr + ' --out $out') #ddg.dot
    elif repr == "json":
        pwd = os.getcwd()
        if out[-4:] != 'json':
            out += '.json'
        joern_process = subprocess.Popen(["./joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
        import_cpg_cmd = f"importCpg(\"{bin}\")\r"
        script_path = f"{pwd}/graph-for-funcs.sc"
        run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{out}\"\r" #json
        cmd = import_cpg_cmd + run_script_cmd
        ret , err = joern_process.communicate(cmd)
        print(ret,err)

    len_outdir = len(glob.glob(outdir + '*'))
    print('--------------> len of outdir ' + repr + '\t' + str(len_outdir))


def main():
    joern_path = '/home/joern-cli_v1.1.172'
    os.chdir(joern_path)
    args = parse_options()
    type = args.type

    input_path = args.input
    output_path = args.output

    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'

    pool_num = 16
    pool = Pool(pool_num)
    
    if type == 'parse':
        files = glob.glob(input_path + '*.c')
        pool.map(partial(joern_parse, outdir = output_path), files)

    elif type == 'export':
        with open(input_path, 'r') as f:
            bins = f.readlines()
        reprs = ['json']
        for repr in reprs:
            print(repr)
            pool.map(partial(joern_export, repr = repr, outdir=output_path), bins)

    else:
        print('Type error!')    

if __name__ == '__main__':
    main()