# Vuldetexp
基于图神经网络的切片级漏洞检测及解释方法

数据集使用的是这篇文章Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). 链接：https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing

数据预处理部分：
首先执行preprocess中的raw_data_preprocess.py读取csv文件获得漏洞减号行相关信息,漏洞减号行格式见nvd_vul_lineinfo.json样例

1.代码标准化：
执行preprocess/code_normalize内的normalization.py文件，注意修改路径

2. Joern生成pdg图和json文件,脚本已经写好在preprocess/joern_graph_gen.py

- -i,-o修改成对应的输入和输出路径
- 首先用-t parse, 生成解析结果.bin文件
- 然后用-t export -r pdg, 生成pdg.dot文件
- 最后用-t export -r lineinfo_json, 生成包含全部cpg信息的lineinfo.json文件

3.生成代码切片
执行preprocess/slice_preocess内的main.py文件，注意输入包括lineinfo.json pdg.dot 以及修改vul_lines字典的路径
输出包括complete_pdg和slice_pdg

4.训练w2v模型并完成嵌入
执行preprocess中的train_w2v.py文件，注意修改路径
执行preprocess中的joern_to_devign文件，完成嵌入，同样注意修改路径

漏洞检测模型训练
打开文件slice_level_model中的main.py 注意修改输入输出路径（包括数据集和模型保存位置） 自己划分train和test（写入两个txt文件） 然后查看确认模型参数无问题运行
这里由于深度学习的不确定性 可能训练的数据和本文有一定出入

漏洞解释模型中解释
这里可以选择我们改进过的GNNExplainer或者PGExplainer GNNExplainer运行benchmar/kernal/pipeline.py,注意参数修改在args.py中,主要的加载部分,首先数据集就是带解释的切片,即数据预处理部分完成节点嵌入的切片;其次是加载模型,路径在args里面设置,选择漏洞检测模型保存的ckpt文件即可,后面注意指定一下输出重要节点的路径.PGE同理不作详细叙述,代码很简单,只需要该路径即可.
interpre_example给出了RQ3中实例的源代码，切片dot文件及解释结果行号

还需要提醒的是,如果有遇到dot文件加载错误的问题,可以执行我们preprocess中的dot_fix.py. 最后解释效果评定代码在preprocess/intrepre_effect.py中,注意需要先执行lineinfo_dict.py从lineinfo中得到对应的行号.

最后版本问题,joern的版本是joern-cli_v1.1.172,可以在 https://github.com/joernio/joern 中查找历史版本下载. networkx版本是2.4或2.5都可以 其他缺少的包根据自己的环境直接pip install安装即可
