import torch
import torch.nn as nn
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.evaluation_utils import evaluate
from config import device

def train_epoch(model, dataloader, criterion, optimizer, use_text, use_image):
    """
    训练一个 epoch。

    参数:
    model (dict): 包含模型组件的字典
    dataloader (DataLoader): 数据加载器
    criterion (nn.Module): 损失函数
    optimizer (torch.optim.Optimizer): 优化器
    use_text (bool): 是否使用文本特征
    use_image (bool): 是否使用图像特征

    返回:
    avg_loss (float): 平均损失
    acc (float): 准确率
    precision (float): 精确率
    recall (float): 召回率
    f1 (float): F1分数
    """
    # 设置每个模型组件为训练模式
    for component in model.values():
        component.train()

    epoch_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch.get('input_ids', None)
        attention_mask = batch.get('attention_mask', None)
        images = batch.get('image', None)
        labels = batch.get('label', None)

        optimizer.zero_grad()

        text_features = None
        image_features = None

        if use_text and input_ids is not None and attention_mask is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            text_features = model['text_encoder'](input_ids, attention_mask)
        if use_image and images is not None:
            images = images.to(device)
            image_features = model['image_encoder'](images)

        labels = labels.to(device)
        logits = model['classifier'](text_features, image_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = epoch_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    return avg_loss, acc, precision, recall, f1


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args, use_text, use_image, epochs):
    """
    训练模型并进行验证，使用早停机制。

    参数:
    model (dict): 包含模型组件的字典
    train_loader (DataLoader): 训练集的数据加载器
    val_loader (DataLoader): 验证集的数据加载器
    criterion (nn.Module): 损失函数
    optimizer (optim.Optimizer): 优化器
    scheduler (lr_scheduler._LRScheduler): 学习率调度器
    args (argparse.Namespace): 命令行参数
    use_text (bool): 是否使用文本特征
    use_image (bool): 是否使用图像特征
    epochs (int): 训练的轮数

    返回:
    best_model (dict): 训练得到的最佳模型组件的字典
    logs (list): 训练日志列表
    train_losses (list): 训练损失列表
    val_losses (list): 验证损失列表
    train_accs (list): 训练准确率列表
    val_accs (list): 验证准确率列表
    train_f1s (list): 训练 F1 分数列表
    val_f1s (list): 验证 F1 分数列表
    """
    print("========== 开始模型训练 ==========")
    best_val_loss = float('inf')  
    best_model = copy.deepcopy(model)
    epochs_no_improve = 0
    logs = []
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, use_text, use_image
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_report = evaluate(
            model, val_loader, criterion, use_text, use_image
        )

        # 根据验证集损失来调整学习率
        scheduler.step(val_loss)  

        log = (f"Epoch {epoch}: "
               f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f} | "
               f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        print(log)
        logs.append(log)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # 检查是否有提升
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("验证集长期无提升，触发早停 (Early Stopping) 机制。")
                break

    return best_model, logs, train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s