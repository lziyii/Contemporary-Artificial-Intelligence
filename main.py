import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import time
import copy
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from models.encoders import TextEncoder, ImageEncoder
from models.classifier import (
    MultiModalGatedFusionClassifier,
    MultiModalConcatenationFusionClassifier,
    MultiModalAttentionFusionClassifier,
    MultiModalLateFusionClassifier
)
from datasets.multimodal_dataset import MultiModalDataset, collate_fn
from utils.data_utils import load_data, calculate_class_weights
from utils.evaluation_utils import evaluate, plot_curves, save_logs
from utils.training_utils import train_model, train_epoch
from utils.testing_utils import test_model
import config


def main(args):
    start_time = time.time()

    experiment_info = []
    experiment_info.append("========== 本次实验使用的参数如下 ==========")
    experiment_info.append(f"学习率 (lr) = {args.lr}")
    experiment_info.append(f"批大小 (batch_size) = {args.batch_size}")
    experiment_info.append(f"训练轮数 (epochs) = {args.epochs}")
    experiment_info.append(f"早停 (patience) = {args.patience}")
    experiment_info.append(f"隐藏层维度 (hidden_dim) = {args.hidden_dim}")
    experiment_info.append(f"Dropout比例 (dropout) = {args.dropout}")
    experiment_info.append(f"是否使用文本特征 (use_text) = {args.use_text}")
    experiment_info.append(f"是否使用图像特征 (use_image) = {args.use_image}")
    experiment_info.append(f"是否处理类别不平衡 (handle_class_imbalance) = {args.handle_class_imbalance}")
    experiment_info.append(f"融合策略 (fusion_type) = {args.fusion_type}")
    experiment_info.append("=========================================\n")

    save_logs(experiment_info, filename='training_log.txt', mode='a')
    print("========== 开始读取数据 ==========")
    train_data, val_data = load_data(config.train_file, config.test_file)

    print("========== 加载 BERT 分词器 ==========")
    tokenizer = BertTokenizer.from_pretrained(config.LOCAL_BERT_PATH)

    # 图像预处理
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    ])

    print("========== 创建Dataset ==========")
    train_dataset = MultiModalDataset(
        train_data, config.data_dir, tokenizer, image_transform,
        use_text=args.use_text, use_image=args.use_image, is_train=True
    )
    val_dataset = MultiModalDataset(
        val_data, config.data_dir, tokenizer, image_transform,
        use_text=args.use_text, use_image=args.use_image, is_train=True
    )

    test_df = pd.read_csv(config.test_file)
    test_dataset = MultiModalDataset(
        test_df, config.data_dir, tokenizer, image_transform,
        use_text=args.use_text, use_image=args.use_image, is_train=False
    )

    print("========== 创建DataLoader ==========")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 计算类别权重
    class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float).to(config.device)
    if args.handle_class_imbalance:
        class_weights = calculate_class_weights(train_data)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.device)

    print("========== 初始化模型 ==========")
    model_components = {}
    if args.use_text:
        model_components['text_encoder'] = TextEncoder(config.LOCAL_BERT_PATH).to(config.device)
    if args.use_text:
        model_components['image_encoder'] = ImageEncoder().to(config.device)

    classifier_input_dim = 0
    if args.use_text:
        classifier_input_dim += 768
    if args.use_image:
        classifier_input_dim += 2048

    if args.fusion_type == 'gate':
        model_components['classifier'] = MultiModalGatedFusionClassifier(
            text_dim=768,
            image_dim=2048,
            hidden_dim=args.hidden_dim,
            num_classes=3,
            dropout=args.dropout,
            use_text=args.use_text,
            use_image=args.use_image
        ).to(config.device)
    elif args.fusion_type == 'concat':
        model_components['classifier'] = MultiModalConcatenationFusionClassifier(
            text_dim=768,
            image_dim=2048,
            hidden_dim=args.hidden_dim,
            num_classes=3,
            dropout=args.dropout,
            use_text=args.use_text,
            use_image=args.use_image
        ).to(config.device)
    elif args.fusion_type == 'attention':
        model_components['classifier'] = MultiModalAttentionFusionClassifier(
            text_dim=768,
            image_dim=2048,
            hidden_dim=args.hidden_dim,
            num_classes=3,
            dropout=args.dropout,
            use_text=args.use_text,
            use_image=args.use_image
        ).to(config.device)
    elif args.fusion_type == 'late':  
        model_components['classifier'] = MultiModalLateFusionClassifier(
            text_dim=768,
            image_dim=2048,
            hidden_dim=args.hidden_dim,
            num_classes=3,
            dropout=args.dropout
        ).to(config.device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([
        {'params': model_components['classifier'].parameters(), 'lr': args.lr}
    ], lr=args.lr)
    if args.use_text:
        model_components['text_encoder'] = TextEncoder(config.LOCAL_BERT_PATH).to(config.device)
    if args.use_image:
        model_components['image_encoder'] = ImageEncoder().to(config.device)

    # 引入动态学习率调度器：根据验证集损失表现衰减学习率
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    print("========== 本次实验使用的参数如下 ==========")
    print(f"学习率 (lr) = {args.lr}")
    print(f"批大小 (batch_size) = {args.batch_size}")
    print(f"训练轮数 (epochs) = {args.epochs}")
    print(f"早停 (patience) = {args.patience}")
    print(f"隐藏层维度 (hidden_dim) = {args.hidden_dim}")
    print(f"Dropout比例 (dropout) = {args.dropout}")
    print(f"是否使用文本特征 (use_text) = {args.use_text}")
    print(f"是否使用图像特征 (use_image) = {args.use_image}")
    print(f"是否处理类别不平衡 (handle_class_imbalance) = {args.handle_class_imbalance}")
    print(f"融合策略 (fusion_type) = {args.fusion_type}")
    print("\n")

    # 训练模型
    best_model, logs, train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model(
        model_components, train_loader, val_loader, criterion, optimizer, scheduler, args, args.use_text, args.use_image, args.epochs
    )

    print("========== 训练完成，正在保存结果 ==========")
    save_logs(logs)
    plot_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, args)

    print("========== 在验证集上使用最佳模型进行评估 ==========")
    _, _, _, _, _, report = evaluate(best_model, val_loader, criterion, args.use_text, args.use_image)
    print("Validation Classification Report:")
    print(report)
    with open('validation_classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存最佳模型
    torch.save(best_model, 'best_multimodal_model.pth')

    print("========== 使用最佳模型在测试集上进行预测 ==========")
    test_df_pred = test_model(best_model, test_loader, args, config)

    # 记录结束时间，在训练完成后，记录结束时间并写入日志
    end_time = time.time()
    total_time = end_time - start_time
    time_info = [f"本次实验总训练时间: {total_time / 60:.2f} 分钟."]

    save_logs(time_info, filename='training_log.txt', mode='a')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--patience', type=int, default=2, help='早停的耐心值')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比例')
    parser.add_argument("--use_text", required=True, type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="是否使用文本特征")
    parser.add_argument("--use_image", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="是否使用图像特征")
    parser.add_argument("--handle_class_imbalance", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="是否处理类别不平衡")
    parser.add_argument("--fusion_type", choices=['gate', 'concat', 'attention', 'late'], default='gate',  
                        help="融合策略，可选 'gate'、'concat'、'attention' 或 'late'")

    args = parser.parse_args()
    main(args)
