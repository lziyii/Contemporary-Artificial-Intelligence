# 多模态情感分析

## 项目概述
本项目旨在设计一个多模态融合模型，该模型利用配对的文本和图像数据预测情感标签。该任务为三分类问题，标签包括：positive、neutral 和 negative。通过对比不同的多模态融合策略，本实验最终选择最佳模型（验证集上的准确率达到70%）并在测试集上进行预测。消融实验进一步验证了多模态数据融合的优势。

## 安装

本实现基于 Python3。要运行代码，你需要以下依赖：

- torch==2.3.1+cu118
- torchaudio==2.3.1+cu118
- torchvision==0.18.1+cu118
- transformers==4.47.1
- pandas==2.2.3
- numpy==1.26.3
- matplotlib==3.10.0
- seaborn==1.3.1
- tqdm==4.67.1
- pillow==11.1.0
- scikit-learn==1.6.0

您可以简单地运行以下命令进行依赖安装：
```python
pip install -r requirements.txt
```


## 仓库结构

以下是本仓库的结构概述：
```python
multimodal_sentiment_analysis/
├── 实验五数据/  # 存储实验数据的目录
│   ├── data/  # 存储所有的训练文本和图像数据，每个文件按照唯一的 guid 命名，包括一个图像和一个文本
│   ├── train.txt  # 训练集文件，包含 4000 个 guid 和对应的情感标签（positive、neutral、negative）
│   └── test_without_label.txt  # 测试集文件，包含数据的 guid 和空的情感标签
│   ├── my_local_bert_base_uncased/ # 存储手动下载的 bert-base-uncased 模型文件，包括 pytorch_model.bin、config.json 和 vocab.txt 等，因为作者遇到 SSL 错误，使用了本地路径加载 BERT 模型，避免网络下载问题。
├── models/  # 包含编码器和分类器的实现
│   ├── __init__.py  # 使 models 文件夹成为一个 Python 包
│   ├── encoders.py  # 使用 BERT 和 ResNet50 的文本和图像编码器
│   └── classifier.py  # 具有不同融合策略（门控、拼接、注意力、晚期融合）的多模态分类器
├── datasets/  # 与数据集相关的代码
│   ├── __init__.py  # 使 datasets 文件夹成为一个 Python 包
│   └── multimodal_dataset.py  # 多模态数据集类，用于数据加载和预处理
├── utils/  # 实用工具函数
│   ├── __init__.py  # 使 utils 文件夹成为一个 Python 包
│   ├── data_utils.py  # 数据加载和类别权重计算
│   ├── evaluation_utils.py  # 评估函数，如分类报告、绘制曲线
│   ├── testing_utils.py  # 测试模型的函数
│   └── training_utils.py  # 训练模型的函数，包括训练一个 epoch 和完整的训练过程
├── main.py  # 主训练脚本
├── config.py  # 配置文件，存储路径和设备设置
├── requirements.txt  # 依赖包列表文件
└── README.md  # 本说明文件

```


## 运行流程

### 1. 数据准备
- 首先，下载`实验五数据`，并从 [huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main) 手动下载模型文件（pytorch_model.bin、config.json、vocab.txt）储存到`my_local_bert_base_uncased`文件夹中。


### 2. 模型训练
- 你可以运行主脚本 `main.py` 来开始训练过程。


以下是运行主脚本的示例，您可以在终端输入：
```python
python main.py --lr 0.0001 --patience 3 --epochs 20 --batch_size 64 --hidden_dim 128 --dropout 0.1 --use_text True --use_image True --handle_class_imbalance False --fusion_type attention
```


### 3. 模型评估
- 训练完成后，脚本将自动在验证集上评估模型，生成分类报告，并将其保存在 `validation_classification_report.txt` 中。它还会绘制训练和验证的损失、准确率和 F1 分数曲线。


### 4. 模型测试
- 训练好的模型将用于预测测试集的情感标签，结果将保存在 `test_prediction.txt` 中。


### 5. 消融实验
- 可以通过修改命令行中的 use_text 和 use_image 参数来执行消融实验，以评估单模态（仅文本或仅图像）和多模态模型。


## 模型亮点

- **多模态融合**：该模型使用预训练的 BERT 进行文本编码，使用 ResNet50 进行图像编码，并尝试各种融合策略（早期拼接、门控、注意力和晚期融合）。
- **数据预处理**：包括对文本（清洗、分词）和图像（调整大小、归一化、数据增强）的全面数据预处理。
- **训练技术**：使用交叉熵损失、Adam 优化器和早停机制来防止过拟合。



## 未来工作

- 进一步优化数据预处理，处理多语言文本。
- 结合OCR技术，分析带有文字的图像（如Meme图像）；引入更复杂的图像分析方法，如姿势估计和表情识别，以提升图像理解能力。



## 许可证

该项目遵循 [MIT 许可证](LICENSE)。


