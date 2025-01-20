import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(train_file, test_file):
    # 读取训练和测试文件
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df['guid'] = train_df['guid'].astype(str).str.replace(r'\.0$', '', regex=True)
    test_df['guid'] = test_df['guid'].astype(str).str.replace(r'\.0$', '', regex=True)

    print("数据读取完成，开始切分训练/验证集...")
    # 分割训练集和验证集
    train_data, val_data = train_test_split(train_df, test_size=0.1, stratify=train_df['tag'], random_state=409)
    return train_data, val_data


def calculate_class_weights(df):
    """
    计算类别权重，缓解类别不平衡问题。
    """
    label_counts = df['tag'].value_counts().to_dict()
    total = sum(label_counts.values())
    class_weights = []
    for label in ['positive', 'neutral', 'negative']:
        if label in label_counts:
            weight = total / (len(label_counts) * label_counts[label])
        else:
            weight = 1.0
        class_weights.append(weight)
    return class_weights
