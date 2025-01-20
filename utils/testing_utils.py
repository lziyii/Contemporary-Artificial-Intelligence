import torch
import pandas as pd


def test_model(best_model, test_loader, args, config):
    """
    使用最佳模型在测试集上进行预测并保存结果。

    参数:
    best_model (dict): 存储最佳模型组件的字典，包括 'text_encoder', 'image_encoder' 和 'classifier'
    test_loader (DataLoader): 测试集的数据加载器
    args (argparse.Namespace): 命令行参数
    config: 配置信息，包含设备信息等
    """
    for component in best_model.values():
        component.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            text_features = None
            image_features = None
            if args.use_text and 'input_ids' in batch and 'attention_mask' in batch:
                input_ids = batch.get('input_ids', None).to(config.device)
                attention_mask = batch.get('attention_mask', None).to(config.device)
                text_features = best_model['text_encoder'](input_ids, attention_mask)
            if args.use_image and 'image' in batch:
                images = batch.get('image', None).to(config.device)
                image_features = best_model['image_encoder'](images)

            logits = best_model['classifier'](text_features, image_features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
            preds = [label_map[p] for p in preds]
            predictions.extend(preds)

    # 保存测试集预测结果
    test_df = pd.read_csv(config.test_file)
    test_df_pred = test_df.copy()
    test_df_pred['tag'] = predictions
    test_df_pred.to_csv('test_prediction.txt', index=False)
    print("测试集预测完成，结果已保存至 test_prediction.txt 文件。")
    return test_df_pred