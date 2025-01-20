import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalGatedFusionClassifier(nn.Module):
    """
    多模态分类器，采用门控机制融合文本特征和图像特征进行分类。
    """
    def __init__(self, text_dim, image_dim, hidden_dim, num_classes, dropout, use_text=True, use_image=True):
        super(MultiModalGatedFusionClassifier, self).__init__()
        self.use_text = use_text
        self.use_image = use_image
        
        input_dim = 0
        if self.use_text:
            input_dim += text_dim
        if self.use_image:
            input_dim += image_dim
        
        # 门控机制的权重参数
        self.text_gate = nn.Linear(text_dim, text_dim) if self.use_text else None
        self.image_gate = nn.Linear(image_dim, image_dim) if self.use_image else None
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_features=None, image_features=None):
        # 如果有文本特征，应用门控加权
        if self.use_text and text_features is not None:
            text_gate_weight = torch.sigmoid(self.text_gate(text_features))
            text_features = text_features * text_gate_weight 

        # 如果有图像特征，应用门控加权
        if self.use_image and image_features is not None:
            image_gate_weight = torch.sigmoid(self.image_gate(image_features))  
            image_features = image_features * image_gate_weight

        # 拼接加权后的文本和图像特征
        combined = []
        if self.use_text and text_features is not None:
            combined.append(text_features)
        if self.use_image and image_features is not None:
            combined.append(image_features)

        # 拼接所有特征
        combined = torch.cat(combined, dim=1)
        logits = self.classifier(combined)
        return logits


class MultiModalConcatenationFusionClassifier(nn.Module):
    """
    多模态分类器，结合文本特征和图像特征进行分类，简单拼接。
    """
    def __init__(self, text_dim, image_dim, hidden_dim, num_classes, dropout, use_text=True, use_image=True):
        super(MultiModalConcatenationFusionClassifier, self).__init__()
        self.use_text = use_text
        self.use_image = use_image
        input_dim = 0
        if self.use_text:
            input_dim += text_dim
        if self.use_image:
            input_dim += image_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_features=None, image_features=None):
        combined = []
        if self.use_text:
            combined.append(text_features)
        if self.use_image:
            combined.append(image_features)
        combined = torch.cat(combined, dim=1)
        logits = self.classifier(combined)
        return logits


class MultiModalAttentionFusionClassifier(nn.Module):
    """
    多模态分类器，采用注意力机制融合文本特征和图像特征进行分类。
    """
    def __init__(self, text_dim, image_dim, hidden_dim, num_classes, dropout, use_text=True, use_image=True):
        super(MultiModalAttentionFusionClassifier, self).__init__()
        self.use_text = use_text
        self.use_image = use_image
        input_dim = 0
        if self.use_text:
            input_dim += text_dim
        if self.use_image:
            input_dim += image_dim

        # 注意力机制的参数
        self.attention_text = nn.Linear(text_dim, 1) if self.use_text else None
        self.attention_image = nn.Linear(image_dim, 1) if self.use_image else None
        
        # 结合文本和图像的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_features=None, image_features=None):
        combined = []
        if self.use_text and text_features is not None:
            combined.append(text_features)
        if self.use_image and image_features is not None:
            combined.append(image_features)

        if len(combined) > 0:
            combined = torch.cat(combined, dim=1)
        else:
            return None
        
        # 计算注意力权重
        if self.use_text and text_features is not None:
            attention_text_weights = torch.softmax(self.attention_text(text_features), dim=1)
            text_features = text_features * attention_text_weights 

        if self.use_image and image_features is not None:
            attention_image_weights = torch.softmax(self.attention_image(image_features), dim=1)
            image_features = image_features * attention_image_weights  

        # 拼接经过注意力加权后的文本和图像特征
        combined = []
        if self.use_text and text_features is not None:
            combined.append(text_features)
        if self.use_image and image_features is not None:
            combined.append(image_features)
        
        combined = torch.cat(combined, dim=1)
        
        # 通过全连接层得到分类结果
        logits = self.classifier(combined)
        return logits


class MultiModalLateFusionClassifier(nn.Module):
    """
    多模态分类器，采用晚期融合，分别对文本和图像特征进行独立分类，再结合分类结果进行最终预测。
    """
    def __init__(self, text_dim, image_dim, hidden_dim, num_classes, dropout):
        super(MultiModalLateFusionClassifier, self).__init__()
        # 文本分类器
        self.text_classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        # 图像分类器
        self.image_classifier = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        # 最终的分类器，将两个分类结果拼接后进行最终分类
        self.final_classifier = nn.Linear(2 * num_classes, num_classes)

    def forward(self, text_features=None, image_features=None):
        # 对文本特征进行分类
        if text_features is not None:
            text_logits = self.text_classifier(text_features)
        else:
            text_logits = None
        # 对图像特征进行分类
        if image_features is not None:
            image_logits = self.image_classifier(image_features)
        else:
            image_logits = None
        # 结合两个分类结果
        if text_logits is not None and image_logits is not None:
            combined_logits = torch.cat((text_logits, image_logits), dim=1)
        elif text_logits is not None:
            combined_logits = text_logits
        elif image_logits is not None:
            combined_logits = image_logits
        else:
            raise ValueError("Both text and image features are None.")
        # 最终分类
        final_logits = self.final_classifier(combined_logits)
        return final_logits
