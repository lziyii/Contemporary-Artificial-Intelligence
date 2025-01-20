import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms


def clean_text(text):
    """
    清洗文本数据，包括替换URL、用户提及，去除#符号，并转为小写。
    """
    # 去除URL
    text = re.sub(r'http\S+|www.\S+', '', text)
    # 去除用户提及
    text = re.sub(r'@\w+', '', text)
    # 去除#符号
    text = re.sub(r'#', '', text)
    # 转为小写
    text = text.lower()
    return text.strip()


class MultiModalDataset(Dataset):
    """
    多模态数据集类，整合文本和图像数据。
    """
    def __init__(self, df, data_dir, tokenizer, transform, use_text=True, use_image=True, is_train=True):
        self.df = df
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.use_text = use_text
        self.use_image = use_image
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        guid = self.df.iloc[idx]['guid']
        guid = str(guid).rstrip('.0')
        features = {}

        if self.use_text:
            # 处理文本
            txt_path = os.path.join(self.data_dir, f"{guid}.txt")
            try:
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception as e:
                print(f"读取文本 {txt_path} 出错: {e}")
                text = ""
            text = clean_text(text)
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=30,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            features['input_ids'] = encoding['input_ids'].squeeze()  
            features['attention_mask'] = encoding['attention_mask'].squeeze()  

        if self.use_image:
            # 处理图像
            img_path = os.path.join(self.data_dir, f"{guid}.jpg")
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"读取图像 {img_path} 出错: {e}")
                # 创建一个全黑的图像 tensor 代替
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            image = self.transform(image)
            features['image'] = image

        if self.is_train:
            label = self.df.iloc[idx]['tag']
            label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
            features['label'] = label_map[label]
            return features
        else:
            features['guid'] = guid
            return features


def collate_fn(batch):
    """
    自定义的 collate 函数，用于将批量数据组织起来。
    """
    batch_features = {}
    if 'input_ids' in batch[0]:
        batch_features['input_ids'] = torch.stack([item['input_ids'] for item in batch])
        batch_features['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])
    if 'image' in batch[0]:
        batch_features['image'] = torch.stack([item['image'] for item in batch])
    if 'label' in batch[0]:
        batch_features['label'] = torch.tensor([item['label'] for item in batch])
    else:
        batch_features['guid'] = [item['guid'] for item in batch]
    return batch_features