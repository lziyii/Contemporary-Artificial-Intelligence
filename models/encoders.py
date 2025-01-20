import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel


class TextEncoder(nn.Module):
    """
    文本编码器，使用预训练的 BERT 模型提取 [CLS] 向量作为文本特征。
    """
    def __init__(self, bert_path):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                         attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :] 
        return cls_output


class ImageEncoder(nn.Module):
    """
    图像编码器，使用预训练的 ResNet50 模型提取图像特征。
    """
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  
        self.resnet = nn.Sequential(*modules)
        self.output_dim = 2048

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)  
        return features