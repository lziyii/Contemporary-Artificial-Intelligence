import torch

data_dir = '实验五数据/data'
train_file = '实验五数据/train.txt'
test_file = '实验五数据/test_without_label.txt'

# 假将 BERT 模型文件放在本地./my_local_bert_base_uncased/ 目录
LOCAL_BERT_PATH = './my_local_bert_base_uncased/'

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前设备: {device}\n")