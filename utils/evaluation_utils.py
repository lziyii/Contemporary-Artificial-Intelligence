import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from config import device


def evaluate(model, dataloader, criterion, use_text, use_image):
    """
    在验证集上评估模型。
    """
    for component in model.values():
        component.eval()

    epoch_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.get('input_ids', None)
            attention_mask = batch.get('attention_mask', None)
            images = batch.get('image', None)
            labels = batch.get('label', None)

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

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = epoch_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=['positive', 'neutral', 'negative'], digits=4)
    return avg_loss, acc, precision, recall, f1, report


import matplotlib.pyplot as plt


def plot_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, args):
    """
    绘制训练和验证的损失曲线及准确率、F1分数曲线。
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 5))
    plt.suptitle(
        f"Training Curves\n"
        f"Parameters: LR={args.lr}, BS={args.batch_size}, EPOCHS={args.epochs}, PATIENCE={args.patience}, "
        f"HIDDEN_DIM={args.hidden_dim}, DROPOUT={args.dropout}, USE_TEXT={args.use_text}, "
        f"USE_IMAGE={args.use_image}, HANDLE_CLASS_IMBALANCE={args.handle_class_imbalance}, "
        f"FUSION_TYPE={args.fusion_type}"
    )

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Acc')
    plt.plot(epochs, val_accs, 'ro-', label='Validation Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # F1分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1s, 'bo-', label='Training F1')
    plt.plot(epochs, val_f1s, 'ro-', label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def save_logs(logs, filename='training_log.txt', mode='a'):
    """
    将训练日志保存到文件，默认以追加方式写入。
    """
    with open(filename, mode, encoding='utf-8') as f:
        for log in logs:
            f.write(log + '\n')